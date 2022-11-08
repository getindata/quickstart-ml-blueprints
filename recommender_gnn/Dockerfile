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

# conda install
ENV CONDA_DIR /opt/conda
COPY install/ubuntu_install_conda.sh /install/ubuntu_install_conda.sh
RUN bash /install/ubuntu_install_conda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda update -y conda
ENV CONDA_ALWAYS_YES="true"

# poetry install
ENV POETRY_VERSION=1.2.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$PATH:${HOME}/.local/bin"
RUN poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./
RUN pip install pip==22.1.2 setuptools==63.1.0

# requirements, we can consider using conda-pack tool with multi-stage builds for smaller image size
COPY environment.yml virtual-packages.yml ./
COPY conda-linux-64.lock ./
RUN conda create --name gnns --file conda-linux-64.lock && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete
SHELL ["conda", "run", "-n", "gnns", "/bin/bash", "-c"]
RUN poetry install && rm -rf ${HOME}/.cache/pypoetry

COPY . .

ENV PATH /opt/conda/envs/gnns/bin:$PATH
ENV CONDA_DEFAULT_ENV gnns

CMD ["kedro", "run", "-p", "santander_e2e"]