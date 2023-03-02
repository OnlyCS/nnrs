FROM ubuntu:20.04

WORKDIR /home/

COPY . .

RUN apt-get update
RUN apt-get install -y \
  curl \
  git \
  gnupg2 \
  jq \
  sudo \
  zsh \
  vim \
  build-essential \
  openssl

RUN useradd -m -s /bin/zsh github
USER github

RUN bash ./setup.sh