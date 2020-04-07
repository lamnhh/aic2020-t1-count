let fs = require("fs");
let path = require("path");

for (let i = 1; i <= 20; ++i) {
  let f = fs.readFileSync(path.join(__dirname, "moi", `cam_${i}.json`));
  let json = JSON.parse(f);

  let mois = json.shapes
    .map(({ label, points }) => ({
      label: parseInt(label),
      points
    }))
    .sort(({ label: label1 }, { label: label2 }) => label1 - label2);

  let ans = {
    width: json.imageWidth,
    height: json.imageHeight,
    mois
  };
  fs.writeFileSync(
    path.join(__dirname, "moi", `cam_${i}_new.json`),
    JSON.stringify(ans)
  );
}
