using CSV, DataFrames
 df = CSV.File("data/exoplanet_archive.csv") |> DataFrame
 println("Columns: ", join(string.(names(df)), ", "))
 println("First pl_name examples: ", df.pl_name[1:5])
