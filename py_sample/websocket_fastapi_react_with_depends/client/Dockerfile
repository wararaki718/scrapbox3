FROM node:15.11.0

WORKDIR /client
COPY . /client

RUN yarn install --production

CMD [ "yarn", "start" ]
