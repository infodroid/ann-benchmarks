# Warning! Do not use this config in production! 
# This is only for testing and security has been turned off.

FROM ann-benchmarks AS builder

ARG SOLR_VERSION="9.7.0"
ARG SOLR_DIST=""
ARG SOLR_DOWNLOAD_SERVER="https://www.apache.org/dyn/closer.lua?action=download&filename=/solr/solr"

RUN apt-get -y install curl

RUN set -ex; \
  apt-get update; \
  apt-get -y --no-install-recommends install wget gpg gnupg dirmngr; \
  rm -rf /var/lib/apt/lists/*; \
  export SOLR_BINARY="solr-$SOLR_VERSION$SOLR_DIST.tgz"; \
  MAX_REDIRECTS=3; \
  case "${SOLR_DOWNLOAD_SERVER}" in \
    (*"apache.org"*);; \
    (*) \
      # If a non-ASF URL is provided, allow more redirects and skip GPG step.
      MAX_REDIRECTS=4 && \
      SKIP_GPG_CHECK=true;; \
  esac; \
  export DOWNLOAD_URL="$SOLR_DOWNLOAD_SERVER/$SOLR_VERSION/$SOLR_BINARY"; \
  echo "downloading $DOWNLOAD_URL"; \
  if ! wget -t 10 --max-redirect $MAX_REDIRECTS --retry-connrefused -nv "$DOWNLOAD_URL" -O "/opt/$SOLR_BINARY"; then rm -f "/opt/$SOLR_BINARY"; fi; \
  if [ ! -f "/opt/$SOLR_BINARY" ]; then echo "failed download attempt for $SOLR_BINARY"; exit 1; fi; \
  tar -C /opt --extract --preserve-permissions --file "/opt/$SOLR_BINARY"; \
  rm "/opt/$SOLR_BINARY"*; \
  apt-get -y remove gpg dirmngr && apt-get -y autoremove;

FROM ann-benchmarks

RUN apt-get -y install curl tmux gosu

# Set the environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install Java
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pysolr==3.10.0 tqdm

COPY --from=builder /opt/solr-9.7.0 /opt/solr

RUN chmod -R 777 /opt/solr

WORKDIR /opt/solr


ENV SOLR_USER="solr" \
    SOLR_UID="8983" \
    SOLR_GROUP="solr" \
    SOLR_GID="8983" \
    PATH="/opt/solr/bin:/opt/solr/docker/scripts:/opt/solr/prometheus-exporter/bin:$PATH" \
    SOLR_INCLUDE=/etc/default/solr.in.sh \
    SOLR_HOME=/var/solr/data \
    SOLR_PID_DIR=/var/solr \
    SOLR_LOGS_DIR=/var/solr/logs \
    LOG4J_PROPS=/var/solr/log4j2.xml \
    SOLR_JETTY_HOST="0.0.0.0" \
    SOLR_ZK_EMBEDDED_HOST="0.0.0.0"

RUN set -ex; \
  groupadd -r --gid "$SOLR_GID" "$SOLR_GROUP"; \
  useradd -r --uid "$SOLR_UID" --gid "$SOLR_GID" "$SOLR_USER" --home /opt/solr -p '*'

# add symlink to /opt/solr, remove what we don't want.
# Remove the Dockerfile because it might not represent the dockerfile that was used to generate the image.
RUN set -ex; \
  (cd /opt; ln -s solr-*/ solr); \
  rm -Rf /opt/solr/docs /opt/solr/docker/Dockerfile;

RUN set -ex; \
  mkdir -p /opt/solr/server/solr/lib /docker-entrypoint-initdb.d; \
  cp /opt/solr/bin/solr.in.sh /etc/default/solr.in.sh; \
  mv /opt/solr/bin/solr.in.sh /opt/solr/bin/solr.in.sh.orig; \
  mv /opt/solr/bin/solr.in.cmd /opt/solr/bin/solr.in.cmd.orig; \
  chmod 0664 /etc/default/solr.in.sh; \
  mkdir -p -m0770 /var/solr; \
  mkdir -p -m0770 /var/solr/data; \
  chown -R "$SOLR_USER:0" /var/solr; \
  test ! -e /opt/solr/modules || ln -s /opt/solr/modules /opt/solr/contrib; \
  test ! -e /opt/solr/prometheus-exporter || ln -s /opt/solr/prometheus-exporter /opt/solr/modules/prometheus-exporter;

RUN set -ex; \
    apt-get update; \
    apt-get -y --no-install-recommends install acl lsof procps wget netcat gosu tini jattach; \
    rm -rf /var/lib/apt/lists/*;

RUN ln -s /opt/solr /home/app/solr

WORKDIR /home/app

EXPOSE 8983

RUN echo 'set -eux' >> entrypoint.sh
RUN echo 'pwd' >> entrypoint.sh
RUN echo 'su - solr -c "nohup /opt/solr/bin/solr start > output.log  2>&1 &"' >> entrypoint.sh
RUN echo 'python3 -u /home/app/run_algorithm.py "$@"' >> entrypoint.sh

ENTRYPOINT ["/bin/bash", "/home/app/entrypoint.sh"]
