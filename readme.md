
Deployment notes: 
I set up SSL to be self-renewing but you have to create the first cert on the deployment host with 
`docker run -it --rm --name certbot -v "/etc/letsencrypt:/etc/letsencrypt" -v "/var/lib/letsencrypt:/var/lib/letsencrypt" certbot/certbot certonly --standalone -d rick-ml.com -d www.rick-ml.com`