

import uvicorn
import traceback
import logging

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from .payloads import CompletionPayload, CompletionResponse, QueueRequest, QueueResponse
from .ops import get_gptj_model

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# globals
MODEL_API = None


@app.on_event("startup")
async def startup_event():
    global MODEL_API
    try:
        MODEL_API = get_gptj_model()
        MODEL_API.load_model()
        MODEL_API.start_background()
    except Exception as e:
        logger.debug(f"Model could not be loaded: {str(e)}")
        traceback.print_exc()


@app.post("/model/get_prediction")
def get_prediction(payload: QueueRequest) -> CompletionResponse:
    res = MODEL_API.wait_for_queue(payload.qid)
    return CompletionResponse(**res)


@app.post("/model/predict")
def model_prediction(payload: CompletionPayload) -> QueueResponse:
    res = MODEL_API.add_to_queue(payload)
    return QueueResponse(qid=res['qid'])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
