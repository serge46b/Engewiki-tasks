const http = require('http');
const fs = require('fs').promises;

const host = "127.0.0.1";
const port = 8080;
const page_dir = "/page"
let index_file;

const requestListener = function (req, res) {
    let req_url = req.url;
    let req_method = req.method;
    console.log("request url: " + req_url);
    console.log("request method: " + req_method);
    if (req_method == "POST") {
        let req_data = '';
        req.on('data', chunk => {
            req_data += chunk.toString();
        });
        req.on('end', () => {
            console.log("requset data: " + req_data);
        });
    }
    if (req_url == "/" && req_method == "GET") {
        res.setHeader("Content-Type", "text/html");
        res.writeHead(200);
        res.end(index_file);
    } else if (req_method == "GET") {
        fs.readFile(__dirname + page_dir + req_url)
            .then(contents => {
                res.setHeader("Content-Type", "text/html");
                res.writeHead(200);
                res.end(contents);
            }).catch(err => {
                console.error("'" + err + "' on file: " + __dirname + page_dir + req_url);
                res.setHeader("Content-Type", "text/html");
                res.writeHead(404);
                res.end("err not found");
            })
    }

}

const server = http.createServer(requestListener);

fs.readFile(__dirname + page_dir + "/index.html")
    .then(contents => {
        index_file = contents;
        server.listen(port, host, () => {
            console.log(`Server is running on http://${host}:${port}/`);
        });
    })
    .catch(err => {
        console.log("'" + err + "' on file: " + __dirname + page_dir + "/index.html");
        process.exit(1);
    });