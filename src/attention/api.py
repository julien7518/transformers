"""uvicorn attention.api:app ; ATTENTION_MODEL_DIR sélectionne le run local."""
from contextlib import asynccontextmanager
import logging
import os
import threading
from typing import Literal
from dataclasses import asdict
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictStr
from attention.inference import Translator, TranslationError, validate_text

logger = logging.getLogger(__name__)

class TranslationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: StrictStr
    direction: Literal['to_genz', 'to_standard']


def error(status, code, message):
    return JSONResponse(status_code=status, content={'error': {'code': code, 'message': message}})


def create_app(model_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.translator = None
        directory = model_dir or os.environ.get('ATTENTION_MODEL_DIR')
        if directory:
            try:
                app.state.translator = Translator(directory)
            except Exception:
                logger.exception('Impossible de charger le modèle local')
        yield
    app = FastAPI(title='Attention — traduction de registre', lifespan=lifespan)
    lock = threading.Lock()

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return error(422, 'INVALID_REQUEST', 'Requête invalide : texte et direction attendus.')

    @app.exception_handler(TranslationError)
    async def translation_error(request, exc):
        return error(422 if exc.code == 'INVALID_REQUEST' else 400, exc.code, str(exc))

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logger.exception('Erreur interne', exc_info=exc)
        return error(500, 'INTERNAL_ERROR', 'Une erreur interne est survenue.')

    @app.get('/api/health')
    def health(request: Request):
        translator = request.app.state.translator
        return {'status': 'ready' if translator else 'unavailable', 'model_id': translator.model_id if translator else None, 'limits': {'max_input_chars': 500, 'max_source_tokens': translator.config.max_length if translator else 64, 'max_new_tokens': translator.max_new_tokens if translator else 63}}

    @app.post('/api/translate')
    def translate(payload: TranslationRequest, request: Request):
        text = validate_text(payload.text, payload.direction)
        translator = request.app.state.translator
        if translator is None:
            return error(503, 'MODEL_UNAVAILABLE', 'Aucun modèle entraîné n’est disponible.')
        with lock:
            return asdict(translator.translate(text, payload.direction))
    return app

app = create_app()
