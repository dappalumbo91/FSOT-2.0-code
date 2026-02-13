using CSV, DataFrames
 df = CSV.File("data/exoplanet_archive.csv") |> DataFrame
 println("Columns: ", names(df))
 println("Sample hostname/pl_name: ", df.hostname[1:5], " / ", df.pl_name[1:5])
