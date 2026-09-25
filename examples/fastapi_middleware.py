"""FastAPI Middleware / Dependency example for open-keypool."""

from fastapi import FastAPI, Depends, HTTPException
import httpx
from open_keypool import AsyncKeyPool, AllKeysExhaustedError

app = FastAPI(title="Open KeyPool FastAPI Integration Example")

# Initialize AsyncKeyPool (e.g. using Groq provider preset)
key_pool = AsyncKeyPool.from_env(suffix="GROQ_KEY", provider="groq")


async def get_groq_completion(prompt: str, key: str) -> dict:
    """Async API call to Groq endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        return response


@app.post("/generate")
async def generate_text(prompt: str):
    """Endpoint utilizing AsyncKeyPool.call() for automatic key rotation."""
    try:
        response = await key_pool.call(get_groq_completion, prompt)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(
            status_code=response.status_code,
            detail=f"API request failed with status {response.status_code}",
        )
    except AllKeysExhaustedError as exc:
        raise HTTPException(
            status_code=429,
            detail=f"Service unavailable: all API keys rate limited. {exc}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
