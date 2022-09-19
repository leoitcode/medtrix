FROM python:3.10

# Configure Poetry
ENV POETRY_VERSION=1.1.13
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install Poetry
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Define PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"
ENV PY_PATH=${POETRY_VENV}/bin/python
ENV PIP_PATH=${POETRY_VENV}/bin/pip

# Copy files to /app
WORKDIR /app
COPY requirements.txt poetry.lock pyproject.toml ./
COPY src/ ./src
COPY scripts/ ./scripts

# Install GenderComputer
WORKDIR /app/scripts/genderComputer/
RUN $PY_PATH setup.py install

# Install incompatible requirements with poetry packages
WORKDIR /app
RUN $PIP_PATH install -r requirements.txt

# Install Poetry packages
RUN $PIP_PATH install requests==2.28.1
RUN poetry config virtualenvs.create false
RUN poetry install

# Install NLTK Packages
RUN $PY_PATH -m nltk.downloader punkt
RUN $PY_PATH -m nltk.downloader wordnet
RUN $PY_PATH -m nltk.downloader omw-1.4

# Streamlit port
EXPOSE 8501

CMD [ "poetry", "run", "streamlit", "run", "src/main.py" ]