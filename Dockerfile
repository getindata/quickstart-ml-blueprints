ARG BASE_IMAGE=nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

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

# python
COPY install/ubuntu_install_conda.sh /install/ubuntu_install_conda.sh
RUN bash /install/ubuntu_install_conda.sh

ENV CONDA_ALWAYS_YES="true"

# poetry
ENV POETRY_VERSION=1.2.0
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$PATH:${HOME}/.local/bin"

# conda
COPY install/ubuntu_setup_conda.sh /install/ubuntu_setup_conda.sh
RUN bash /install/ubuntu_setup_conda.sh

COPY . .

CMD ["kedro", "run"]
