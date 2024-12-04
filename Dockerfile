FROM python:3.11-slim

WORKDIR /app

COPY ./ ./

RUN pip install poetry
RUN poetry install

EXPOSE 8501

CMD ["poetry","run","streamlit","run","./fashion_mnist_classifier_2_0/app.py"]

