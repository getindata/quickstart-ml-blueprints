ARG BASE_IMAGE=python:3.8.12-slim

FROM $BASE_IMAGE
# Change the above line for the one below to enable emulated ARM64 (Apple M1/M2) support
# FROM --platform=linux/amd64 $BASE_IMAGE

RUN apt-get update && apt-get upgrade -y

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Europe/Warsaw apt-get install -y git make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.3.2
ADD https://install.python-poetry.org/ /tmp/poetry-installer
RUN python3 /tmp/poetry-installer --version $POETRY_VERSION
ENV PATH="${POETRY_HOME}/bin:$PATH"

# requirements
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./
RUN poetry run pip install --upgrade pip
RUN poetry install && rm -rf ${HOME}/.cache/pypoetry

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

# copy the whole project except what is in .dockerignore
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

CMD ["kedro", "run"]
