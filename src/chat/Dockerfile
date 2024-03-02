FROM python:3.11-slim

RUN pip install poetry==1.8.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi

ARG PORT

EXPOSE ${PORT:-8501}

ENTRYPOINT ["streamlit", "run", "./app/main.py"]