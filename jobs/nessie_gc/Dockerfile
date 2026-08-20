FROM eclipse-temurin:21-jre

# Pinned to the running Nessie server version: the GC tool walks the version store,
# so client and server should move together.
ARG NESSIE_VERSION=0.108.4

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/nessie-gc

RUN curl -fsSL -o nessie-gc.jar \
      "https://github.com/projectnessie/nessie/releases/download/nessie-${NESSIE_VERSION}/nessie-gc-${NESSIE_VERSION}.jar"

COPY run_gc.sh .

RUN chmod +x run_gc.sh && \
    useradd --create-home gc && \
    chown -R gc:gc /opt/nessie-gc

USER gc

ENTRYPOINT ["/opt/nessie-gc/run_gc.sh"]
