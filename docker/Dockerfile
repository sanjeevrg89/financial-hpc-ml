FROM rayproject/ray:latest

WORKDIR /app

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY requirements.txt /app/

RUN pip install -r requirements.txt

CMD ["python3"]