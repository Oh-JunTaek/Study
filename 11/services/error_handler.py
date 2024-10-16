import logging
from flask import jsonify
import openai
from openai import OpenAIError

# 로깅 설정
logging.basicConfig(
    filename='./logging/error_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found_error(e):
        return jsonify({
            'error': "Resource not found",
            'code': 404,
            'description': str(e)
        })

    @app.errorhandler(400)
    def bad_request_error(e):
        return jsonify({
            'error': "Bad request",
            'code': 400,
            'description': str(e)
        })

    @app.errorhandler(500)
    def internal_error(e):
        logging.error(f"500 Internal Server Error: {str(e)}")
        return jsonify({
            'error': "Internal server error",
            'code': 500,
            'description': str(e)
        })

    @app.errorhandler(OpenAIError)
    def handle_openai_error(e):
        if isinstance(e, openai.BadRequestError):
            return jsonify({
                'error': "BadRequestError",
                'code': 400,
                'description': "죄송해요. 시스템 오류가 발생했어요. 잠시 후 다시 시도해주세요.",
                'error_message': str(e)
            })
        elif isinstance(e, openai.AuthenticationError):
            return jsonify({
                'error': "AuthenticationError",
                'code': 401,
                'description': "OpenAI 인증에 실패했습니다. 관리자에게 API 키를 확인해주세요.",
                'error_message': str(e)
            })
        elif isinstance(e, openai.PermissionDeniedError):
            return jsonify({
                'error': "PermissionDeniedError",
                'code': 403,
                'description': "OpenAI API가 지원되지 않는 국가에서 요청하고 있습니다. 죄송해요, 서비스 이용이 불가합니다.",
                'error_message': str(e)
            })
        elif isinstance(e, openai.NotFoundError):
            return jsonify({
                'error': "NotFoundError",
                'code': 404,
                'description': "OpenAI API에서 요청한 자원을 찾을 수 없습니다.",
                'error_message': str(e)
            })
        elif isinstance(e, openai.UnprocessableEntityError):
            return jsonify({
                'error': "UnprocessableEntityError",
                'code': 422,
                'description': "시스템 오류가 발생했습니다. 관리자에게 연락해주세요.",
                'error_message': str(e)
            })
        elif isinstance(e, openai.RateLimitError):
            return jsonify({
                'error': "Rate limit exceeded",
                'code': 429,
                'description': "요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
                'error_message': str(e)
            })
        elif isinstance(e, openai.InternalServerError):
            return jsonify({
                'error': "API connection failed",
                'code': 500,
                'description': "내부 서버 오류로 OpenAI API 연결에 실패했습니다. 잠시 후 다시 시도해주세요.",
                'error_message': str(e)
            })
        else:
            return jsonify({
                'error': "Unknown OpenAI error",
                'code': 500,
                'description': "알 수 없는 OpenAI API 오류가 발생했습니다.",
                'error_message': str(e)
            })

    @app.errorhandler(TypeError)
    def handle_type_error(e):
        return jsonify({
            'error': "TypeError",
            'code': 400,
            'description': "잘못된 데이터 형식이 전달되었습니다.",
            'error_message': str(e)
        })



    