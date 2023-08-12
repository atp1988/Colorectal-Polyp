
FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./server.py /code/server.py
COPY ./src/net.py /code/net.py
COPY ./src/utils.py /code/utils.py
COPY ./src/modules.py /code/modules.py
COPY ./src/pvt.py /code/pvt.py
COPY ./checkpoints/ckpt_pvt2_Decoder_2.pth /code/ckpt_pvt2_Decoder_2.pth 

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
