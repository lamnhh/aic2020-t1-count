let fs = require("fs");
let path = require("path");

for (let i = 1; i <= 20; ++i) {
  let f = fs
    .readFileSync(path.join(__dirname, "roi", `cam_${i}.txt`))
    .toString();
  let points = f
    .split("\r\n")
    .slice(0, -1)
    .map((str) => str.split(",").map((a) => parseInt(a)));
  fs.writeFileSync(
    path.join(__dirname, "roi", `cam_${i}_new.json`),
    JSON.stringify(points)
  );
}
