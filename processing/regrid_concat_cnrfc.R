# Reproject and concatenate raw CNRFC 6-hourly precip into a file

library(raster)
library(ncdf4)
# library(sf)
library(rgdal)

# read in basins/states for sanity checks
basins <- shapefile('/media/kden/LaCie/DATA/geog/sierra_hwbasins/sierra_hwbasins.shp')
states <- shapefile('/media/kden/LaCie/DATA/geog/states/states_wgs.shp')


# function to project and reorient raw CNRFC to WGS coords
# and align with other vector data (since axes were flipped/transposed)
reorient_cnrfc <- function(fname) {
  f <- raster(fname)
  crs(f) <- "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +k=1 +x_0=2466975 +y_0=6534150 +a=6371200 +b=6371200 +to_meter=4762.5 +no_defs"
  fproj <- projectRaster(f, crs=crs(basins))
  # fout <- flip(flip(t(fproj),'x'),'y')
  # return(fout)
  return(fproj)
}

# switch directory to where the files are
setwd('/media/kden/hdd1/phd2_RosDiffs/QPE/')



# create a list of dates to correspond file names
# RECALL THESE REFER TO THE END OF THE ACCUMULATION PERIOD
# d1 <- as.POSIXct('2017-01-01 06:00:00', tz='UTC')
d1 <- as.POSIXct('2016-11-01 06:00:00', tz='UTC')   
# d2 <- as.POSIXct('2017-05-01 00:00:00', tz='UTC')
d2 <- as.POSIXct('2017-05-07 00:00:00', tz='UTC')

dates <- seq(d1, d2, by='6 hours')
# ^ including Mar 1 00 because it refers to the end of accumulation period

# given a date, generate a filename
idate <- dates[1]

# function to generate a file name given date object
get_cnrfc_filename <- function(idate) {
  # idate -- posix datetime object, should be in utc
  monthyear_dir <- strftime(idate, format='%b%Y', tz='UTC')
  fname_base <- strftime(idate, format='qpe.%Y%m%d_%H00.nc', tz='UTC')
  fname <- paste0('./', monthyear_dir, '/', fname_base)
  return(fname)
}


# routine to parse through date times, 
# process each file for each date, and create a 
# raster stack to save to one file --> qpe_janfeb2017_6hr.tif
# each timestep should match the file name == end of 6h accum period
cnrfc_stack <- list()
for (i in 1:(length(dates))) {
  idate <- dates[i]
  fname <- get_cnrfc_filename(idate)
  tmp <- reorient_cnrfc(fname)
  names(tmp) <- strftime(idate, format='%Y-%m-%d %H00', tz='UTC')
  cnrfc_stack[[i]] <- tmp
  cat(paste0('\n --- processed ', strftime(idate, format='%d-%b %H:00', tz='UTC'),' --- \n'))
}
cnrfc <- brick(cnrfc_stack)
# ---
cnrfc

# save
# ofname <- 'qpe_janfeb2017_6hr.tif'
ofname <- 'qpe_wy2017_6hr.tif'

outfile <- writeRaster(cnrfc, filename=ofname, 
                       format="GTiff", overwrite=TRUE)

# --- test re-reading
tmp <- brick(ofname)   # <-- we lose the layer nmes (i.e. time steps)

# that's fine, just keep the date range consistent











