// This is the Javascript to download NAIP image from Google Earth Engine

// Boundary Data

var citybounds = ee.FeatureCollection('Name of Folder to save Image');

Map.addLayer(citybounds, {color: 'blue'}, 'City Boundary');


// Get the bounding box (rectangle) of the citybounds
var bbox = citybounds.geometry().bounds();

Map.addLayer(bbox, {color: 'red'}, 'City Box');


// Show it in the map
// Map.addLayer(citybounds, {}, 'City Boundary');

// NAIP Images
var naip = ee.ImageCollection('USDA/NAIP/DOQQ')
                  .filter(ee.Filter.date('2021-01-01', '2021-12-31'))
                  .mosaic()
                  .clip(bbox)
                  // .reproject(ee.Projection('EPSG:26915'),null,1);
                  
var visParams = {
  bands: ['R', 'G', 'B'],
  min: 0,
  max: 255
};

Map.addLayer(naip, visParams, 'NAIP Image');

var projection = naip.select('N').projection().getInfo();
print(projection)
  Export.image.toDrive({
    image: naip,
    description: "StPaul_2021",
    scale: 1,
    maxPixels: 10000000000000,
    crs: 'EPSG:26915',
    region: citybounds
  });
