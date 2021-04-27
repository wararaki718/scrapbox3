FROM docker.elastic.co/elasticsearch/elasticsearch:7.7.1

RUN yum install -y wget unzip
RUN bin/elasticsearch-plugin install https://github.com/WorksApplications/elasticsearch-sudachi/releases/download/v2.1.0-es7.4-7.7/analysis-sudachi-7.7.1-2.1.0.zip

RUN wget http://sudachi.s3-website-ap-northeast-1.amazonaws.com/sudachidict/sudachi-dictionary-20201223-core.zip && \
    unzip sudachi-dictionary-20201223-core.zip && \
    mkdir config/sudachi && \
    mv sudachi-dictionary-20201223/system_core.dic config/sudachi/ && \
    rm -rf sudachi-dictionary-20201223
