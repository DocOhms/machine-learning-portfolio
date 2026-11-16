const path = require('path');
// Optional raster export uses the separately installed Node.js sharp package.
const sharp = require('sharp');
(async () => {
  for (const name of ['fig10','fig12','fig13','fig14','fig15']) {
    await sharp(path.join(__dirname,'map_sources',name+'.svg'), {density: 240})
      .png().toFile(path.join(__dirname,'figures',name+'.png'));
    console.log('Rendered '+name);
  }
})().catch(err=>{console.error(err);process.exit(1)});
