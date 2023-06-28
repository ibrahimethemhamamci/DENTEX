FROM python:3.8


RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"


RUN python -m pip install --user -U pip && python -m pip install --user pip-tools


COPY --chown=user:user requirements.txt /opt/app/

RUN python -m piptools sync requirements.txt

COPY --chown=user:user ground-truth /opt/app/ground-truth

COPY --chown=user:user test /opt/app/test


COPY --chown=user:user evaluation.py /opt/app/

VOLUME /opt/app/test

ENTRYPOINT [ "python", "-m", "evaluation" ]


