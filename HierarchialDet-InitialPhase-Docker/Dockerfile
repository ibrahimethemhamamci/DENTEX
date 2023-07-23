FROM python:3.8

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --upgrade pip
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools


COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt

#COPY --chown=user:user detectron2 /opt/app/detectron2
#COPY --chown=user:user hierarchialdet /opt/app/hierarchialdet
#COPY --chown=user:user configs /opt/app/configs
#COPY --chown=user:user pretrained_model /opt/app/pretrained_model

COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
