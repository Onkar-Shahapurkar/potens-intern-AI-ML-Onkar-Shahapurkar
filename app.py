from fastapi import FastAPI

app = FastAPI(
    title="POTENS AI/ML Assignment",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "POTENS AI/ML Assignment Backend"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }