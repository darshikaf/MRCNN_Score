FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

ENV CONDA_HOME=/opt/conda
ENV PATH=$CONDA_HOME/bin:$PATH

RUN apt-get update \
&& apt-get install -y \
&& rm -rf /var/lib/apt/lists/* /tmp/*

COPY ./app /app

COPY ./libs /
COPY ./requirements.txt /

RUN pip install /mask_rcnn-2.1-py3-none-any.whl
RUN pip install -r /requirements.txt
