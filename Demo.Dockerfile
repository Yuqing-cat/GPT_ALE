

# Stage 1: build frontend ui
FROM mcr.microsoft.com/cbl-mariner/base/nodejs:16 as ui-build

COPY ./gpt_ale_ui /usr/src/gpt_ale_ui

WORKDIR /usr/src/gpt_ale_ui 

RUN npm install && npm run build

# Stage 2: build backend and start nginx to as reserved proxy for both ui and backend

FROM mcr.microsoft.com/devcontainers/python:dev-3.9

COPY ./gpt_ale_api /usr/src/gpt_ale_api

WORKDIR /usr/src/gpt_ale_api

RUN pip install -r requirements.txt

RUN apt-get update -y && apt-get install -y nginx openssh-server

COPY ./deploy/nginx.conf /etc/nginx/nginx.conf
COPY --from=ui-build /usr/src/gpt_ale_ui/build /usr/share/nginx/html

RUN echo "root:Docker!" | chpasswd 

# Copy the sshd_config file to the /etc/ssh/ directory
COPY sshd_config /etc/ssh/

# Copy and configure the ssh_setup file
RUN mkdir -p /tmp
COPY ssh_setup.sh /tmp
RUN chmod +x /tmp/ssh_setup.sh \
    && (sleep 1;/tmp/ssh_setup.sh 2>&1 > /dev/null)

# Open port 2222 for SSH access
EXPOSE 80 2222 8000

# Start web server
COPY ./deploy/start.sh .

RUN ["chmod", "+x", "./start.sh"]
CMD ["/bin/sh", "-c", "./start.sh"]
