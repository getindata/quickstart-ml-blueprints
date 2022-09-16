ARG BASE_IMAGE=nvidia/cuda:11.3.0-cudnn8-devel-ubuntu22.04

FROM $BASE_IMAGE

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


# pyenv
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

# python
ENV PYTHON_VERSION=3.8.12
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

# poetry
ENV POETRY_VERSION=1.2.0
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$PATH:${HOME}/.local/bin"

# requirements
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./
RUN poetry run pip install --upgrade pip
# RUN pip install wheel==0.37.1
RUN poetry install
#RUN poetry run poe force-torch
RUN python -m poethepoet force-torch
#RUN poetry run poe force-dgl
RUN python -m poethepoet force-dgl
RUN rm -rf ${HOME}/.cache/pypoetry
RUN pyenv rehash

COPY . .

CMD ["ls"]
