FROM python:3.11-slim

RUN apt-get update && apt-get install wkhtmltopdf -y

RUN pip install poetry==1.8.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

RUN poetry install --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi

ARG PORT

EXPOSE ${PORT:-9500}

WORKDIR ./app

CMD [ "waitress-serve", "--port", "9500", "--call", "main:create_app" ]