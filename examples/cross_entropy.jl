using CSV
using DataFrames
using OptimalTransportDataIntegration


params = DataParameters(nA = 1000, nB = 1000)
    
for i in 1:10

    data = generate_xcat_ycat(params)
        
    yb, za = otrecod(data, OTjoint())
    ot = accuracy(data, yb, za)
    yb, za = otrecod(data, SimpleLearning())
    sl = accuracy(data, yb, za)
    yb, za = otrecod(data, UnbalancedModality())
    ote = accuracy(data, yb, za)

    println( " OT : $ot \t SL : $sl \t OTE : $ote ")

end
