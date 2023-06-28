ARG BASE_IMAGE=nvidia/cuda:11.3.0-runtime-ubuntu20.04

FROM $BASE_IMAGE
# Change the above line for the one below to enable emulated ARM64 (Apple M1/M2) support
# FROM --platform=linux/amd64 $BASE_IMAGE

RUN apt-get update && apt-get upgrade -y

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Europe/Warsaw apt-get install -y git make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# add kedro user
ARG KEDRO_UID=1000
ARG KEDRO_GID=1000
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -d /home/kedro -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro

# copy the whole project except what is in .dockerignore
ENV HOME /home/kedro
WORKDIR ${HOME}
RUN chown -R kedro:${KEDRO_GID} ${HOME}
USER kedro
RUN chmod -R a+w ${HOME}

# pyenv
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

# python
ENV PYTHON_VERSION=3.10.10
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

# poetry
ENV POETRY_VERSION=1.3.2
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$PATH:${HOME}/.local/bin"

# requirements
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./
RUN poetry run pip install --upgrade pip
RUN poetry install && rm -rf ${HOME}/.cache/pypoetry
RUN pyenv rehash

COPY . .

CMD ["kedro", "run"]
