using CSV
using DataFrames
using OptimalTransportDataIntegration


params = DataParameters(nA = 1000, nB = 1000)
    
for i in 1:10

    data = generate_xcat_ycat(params)
        
    yb_ot, za_ot = otrecod(data, OTjoint())
    ot = accuracy(data, yb_ot, za_ot)
    yb_ote, za_ote = otrecod(data, UnbalancedModality())
    ote = accuracy(data, yb_ote, za_ote)
    yb_sl, za_sl = otrecod(data, SimpleLearning())
    sl = accuracy(data, yb_sl, za_sl)

    println( " OT : $ot \t SL : $sl \t OTE : $ote ")

end
