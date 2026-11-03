from contextlib import asynccontextmanager
import io
import os
import threading
import asyncio
import time
from typing import List, Optional, Union
import numpy as np
from fastapi import FastAPI, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse, Response
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionInfo, Segment, get_compression_ratio, get_suppressed_tokens
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps
from pydantic import BaseModel
from urllib.parse import urlparse
import aiohttp
from dotenv import load_dotenv

load_dotenv()

model: Optional[WhisperModel] = None
CONCURRENCY = int(os.getenv("TRANSCRIBE_CONCURRENCY", os.getenv("CONCURRENCY", "2")))
transcribe_semaphore = asyncio.Semaphore(CONCURRENCY)
MAX_CLIP_BYTES = 25 * 1024 * 1024
http_session: Optional[aiohttp.ClientSession] = None
# Encode and decode all clips of a request as one batch. Off until the batched output has been
# checked against the per-clip path on a GPU.
BATCH_TRANSCRIBE = os.getenv("BATCH_TRANSCRIBE", "0").lower() in ("1", "true", "yes")

async def load_model():
    global model
    model_name = os.getenv("MODEL_NAME", "turbo")
    device_type = os.getenv("DEVICE_TYPE", "cuda")  # "cpu", "cuda", "auto"
    compute_type = os.getenv("COMPUTE_TYPE", "float16")  # https://opennmt.net/CTranslate2/quantization.html

    # CTranslate2 CPU threading: split cores evenly across concurrent workers.
    # num_workers = number of concurrent inference slots in CTranslate2's internal pool.
    # cpu_threads = threads per inference call (intra-op parallelism).
    num_cores = os.cpu_count() or 4
    ct2_num_workers = CONCURRENCY
    ct2_cpu_threads = max(1, num_cores // ct2_num_workers)

    print(f"Loading model: {model_name} (device={device_type}, compute={compute_type}, "
          f"cpu_threads={ct2_cpu_threads}, num_workers={ct2_num_workers})...")
    try:
        model = WhisperModel(
            model_name,
            device=device_type,
            compute_type=compute_type,
            cpu_threads=ct2_cpu_threads,
            num_workers=ct2_num_workers,
        )
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise ValueError(f"Failed to load model {model_name}: {e}") from e

def _redact_url(url: str) -> str:
    # Tempstore URLs can carry access tokens in the query string; keep them out of logs.
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

async def download_clip(url: str) -> io.BytesIO:
    """
    Downloads an HTTP/HTTPS audio URL into memory.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Unsupported audio URL scheme: {parsed.scheme!r}")
    if http_session is None:
        raise RuntimeError("HTTP session not initialized")

    async with http_session.get(url) as resp:
        resp.raise_for_status()
        if resp.content_length is not None and resp.content_length > MAX_CLIP_BYTES:
            raise ValueError(f"Clip too large: {resp.content_length} bytes")
        buf = io.BytesIO()
        async for chunk in resp.content.iter_chunked(65536):
            buf.write(chunk)
            if buf.tell() > MAX_CLIP_BYTES:
                raise ValueError(f"Clip exceeds {MAX_CLIP_BYTES} bytes")
    buf.seek(0)
    return buf

def _run_transcribe_sync(audio_file: Union[io.BytesIO, np.ndarray]) -> tuple[List[Segment], TranscriptionInfo]:
    global model
    if model is None:
        raise RuntimeError("Model not loaded")
    segments, info = model.transcribe(
        audio_file,
        task="transcribe",
        # language=None would run detect_language(), which encodes the audio once, and then
        # generate_segments() encodes it again. "en" is only a placeholder that skips that
        # first pass: multilingual=True detects the language from each 30s segment's encoder
        # output and swaps the prompt's language token before decoding, so non-English speech
        # is still transcribed in its own language. info.language will always read "en".
        language="en",
        multilingual=True,
        # log_progress=True,
        beam_size=1,
        # best_of=5,
        # patience=1,
        # length_penalty=None,
        # temperature=tuple(np.arange(0, 1.0 + 1e-6, 0.2)),
        # compression_ratio_threshold=2.4,
        # log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        # condition_on_previous_text=True,
        suppress_blank=True,
        # suppress_tokens=[-1],
        without_timestamps=True,
        # max_initial_timestamp=1.0,
        word_timestamps=False,
        vad_filter=True,
    )
    return list(segments), info

# Decoding thresholds from model.transcribe's defaults, mirrored by the batched path.
NO_SPEECH_THRESHOLD = 0.6
LOG_PROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4

def _run_batch_sync(buffers: List[io.BytesIO]) -> List[Union[dict, BaseException, np.ndarray]]:
    """
    Transcribes clips of 30s or less (after VAD) with one batched encode and one batched generate,
    reproducing what model.transcribe does for a single 30s window with the settings in
    _run_transcribe_sync. Each result is a dict (text, duration, output_tokens), the exception
    for that clip, or the decoded audio when the clip must go through model.transcribe instead
    (longer than 30s, or greedy decoding would trigger temperature fallback).
    """
    global model
    if model is None:
        raise RuntimeError("Model not loaded")
    fe = model.feature_extractor
    results: List = [None] * len(buffers)
    batch_idx, batch_features, batch_audio = [], [], []

    for i, buf in enumerate(buffers):
        try:
            audio = decode_audio(buf, sampling_rate=fe.sampling_rate)
            duration = audio.shape[0] / fe.sampling_rate
            audio_chunks, _ = collect_chunks(audio, get_speech_timestamps(audio, VadOptions()))
            features = fe(np.concatenate(audio_chunks, axis=0))
            # Same slicing as generate_segments: the last frame is not part of the content.
            content_frames = features.shape[-1] - 1
            if content_frames == 0:
                results[i] = {"text": "", "duration": duration, "output_tokens": 0}
            elif content_frames > fe.nb_max_frames:
                results[i] = audio
            else:
                results[i] = {"duration": duration}
                batch_idx.append(i)
                batch_features.append(pad_or_trim(features[:, :content_frames]))
                batch_audio.append(audio)
        except Exception as e:
            results[i] = e

    if not batch_idx:
        return results

    encoder_output = model.encode(np.stack(batch_features))

    multilingual = model.model.is_multilingual
    if multilingual:
        language_tokens = [r[0][0] for r in model.model.detect_language(encoder_output)]
    else:
        language_tokens = [None] * len(batch_idx)

    tokenizers, prompts = [], []
    for language_token in language_tokens:
        # Built like generate_segments does with multilingual=True: placeholder "en", then swapped.
        tokenizer = Tokenizer(model.hf_tokenizer, multilingual, task="transcribe", language="en")
        if language_token is not None:
            tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
            tokenizer.language_code = language_token[2:-2]
        tokenizers.append(tokenizer)
        prompts.append(model.get_prompt(tokenizer, [], without_timestamps=True))

    generated = model.model.generate(
        encoder_output,
        prompts,
        beam_size=1,
        patience=1,
        length_penalty=1,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        max_length=model.max_length,
        return_scores=True,
        return_no_speech_prob=True,
        suppress_blank=True,
        suppress_tokens=get_suppressed_tokens(tokenizers[0], [-1]),
        max_initial_timestamp_index=int(round(1.0 / model.time_precision)),
    )

    for i, audio, tokenizer, result in zip(batch_idx, batch_audio, tokenizers, generated):
        tokens = result.sequences_ids[0]
        avg_logprob = result.scores[0] * len(tokens) / (len(tokens) + 1)
        text = tokenizer.decode(tokens)
        no_speech = result.no_speech_prob > NO_SPEECH_THRESHOLD
        low_logprob = avg_logprob < LOG_PROB_THRESHOLD
        # Same rules as generate_with_fallback and generate_segments at temperature 0.
        needs_fallback = (
            get_compression_ratio(text.strip()) > COMPRESSION_RATIO_THRESHOLD or low_logprob
        ) and not (no_speech and low_logprob)
        if needs_fallback:
            results[i] = audio
        elif (no_speech and not avg_logprob > LOG_PROB_THRESHOLD) or not text.strip():
            results[i].update(text="", output_tokens=0)
        else:
            results[i].update(text=text.lstrip(), output_tokens=len(tokens))

    return results

async def _transcribe_audio(audio: Union[io.BytesIO, np.ndarray]) -> "TranscriptionClipResponse":
    start_time = time.time()
    async with transcribe_semaphore:
        segments, info = await asyncio.to_thread(_run_transcribe_sync, audio)
    transcription_time = time.time() - start_time
    text = " ".join([segment.text.lstrip() for segment in segments])
    output_tokens = sum(len(segment.tokens) for segment in segments)
    print(f"Transcribed {info.duration}s clip, took {transcription_time}s, {output_tokens} tokens")
    return TranscriptionClipResponse(text=text, duration=info.duration, output_tokens=output_tokens, transcription_time=transcription_time)

async def transcribe_clip(url: str) -> "TranscriptionClipResponse":
    return await _transcribe_audio(await download_clip(url))

async def transcribe_batch(urls: List[str]) -> list:
    """
    Returns one TranscriptionClipResponse or exception per URL, in input order.
    """
    outcomes: list = await asyncio.gather(*(download_clip(url) for url in urls), return_exceptions=True)
    pending = [i for i, d in enumerate(outcomes) if not isinstance(d, BaseException)]
    if not pending:
        return outcomes

    start_time = time.time()
    async with transcribe_semaphore:
        batch_results = await asyncio.to_thread(_run_batch_sync, [outcomes[i] for i in pending])
    transcription_time = time.time() - start_time
    print(f"Batch-transcribed {len(pending)} clips, took {transcription_time}s")

    single = []
    for i, r in zip(pending, batch_results):
        if isinstance(r, dict):
            outcomes[i] = TranscriptionClipResponse(**r, transcription_time=transcription_time)
        else:
            outcomes[i] = r
            if isinstance(r, np.ndarray):
                single.append(i)

    single_results = await asyncio.gather(*(_transcribe_audio(outcomes[i]) for i in single), return_exceptions=True)
    for i, r in zip(single, single_results):
        outcomes[i] = r
    return outcomes

def _start_loader_thread():
    try:
        asyncio.run(load_model())
    except Exception as e:
        print(f"Background model loader failed: {e}")

threading.Thread(target=_start_loader_thread, daemon=True).start()

@asynccontextmanager
async def lifespan(_: FastAPI):
    global http_session, model
    http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    try:
        yield
    finally:
        await http_session.close()
        http_session = None

    if model:
        model = None

app = FastAPI(title="Whisper Load Balancer", lifespan=lifespan)

class TranscriptionRequest(BaseModel):
    audio_buffers: list[str]

class TranscriptionClipResponse(BaseModel):
    text: str
    duration: float
    output_tokens: int
    transcription_time: float

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """
    If API_KEY env var is set, require the same key in the Authorization header.
    Accepts the raw key or 'Bearer <key>'.
    """
    api_key = os.getenv("API_KEY")
    # If no API_KEY is configured, skip verification
    if not api_key:
        return

    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    # Accept the raw key or "Bearer <key>"
    if authorization == api_key:
        return
    if authorization.startswith("Bearer ") and authorization.split(" ", 1)[1] == api_key:
        return

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

request_count = 0

@app.get("/ping", dependencies=[Depends(verify_api_key)])
async def health_check():
    if model is None:
        # 204 responses must not carry a body.
        return Response(status_code=204)

    return {"status": "healthy"}

@app.post("/transcribe", dependencies=[Depends(verify_api_key)])
async def transcribe(request: TranscriptionRequest):
    global request_count, model
    request_count += 1

    if model is None:
        return JSONResponse(
            content={
                "error": "ServiceUnavailable",
                "message": "Model not ready"
            },
            status_code=503
        )

    start_time = time.time()
    if BATCH_TRANSCRIBE:
        outcomes = await transcribe_batch(request.audio_buffers)
    else:
        tasks = [transcribe_clip(url) for url in request.audio_buffers]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for url, outcome in zip(request.audio_buffers, outcomes):
        if isinstance(outcome, BaseException):
            if isinstance(outcome, aiohttp.ClientResponseError):
                reason = f"HTTP {outcome.status} {outcome.message}"
            else:
                reason = repr(outcome).replace(url, _redact_url(url))
            print(f"Clip {_redact_url(url)} failed: {reason}")
            results.append({"text": "", "duration": 0, "output_tokens": 0, "transcription_time": 0, "error": True})
        else:
            results.append(outcome.model_dump())
    request_time = time.time() - start_time
    print(f"Request with {len(results)} clips took {request_time}s")
    return {"results": results}

# A simple endpoint to show request stats
@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def stats():
    return {"total_requests": request_count}

# Run the app when the script is executed
if __name__ == "__main__":
    import uvicorn

    # When you deploy the endpoint, make sure to expose port 5000
    # And add it as an environment variable in the Runpod console
    port = int(os.getenv("PORT", "5000"))

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)
