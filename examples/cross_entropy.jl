using CSV
using DataFrames
using OptimalTransportDataIntegration


mB_values = ([0,0,0], [1,0,0], [1,1,0] , [1,2,0])

for mB = mB_values

    params = DataParameters(nA = 1000, nB = 1000, mB = mB)
        
    data = generate_xcat_ycat(params)

        
    yb_ot, za_ot = otrecod(data, OTjoint())
    ot = accuracy(data, yb_ot, za_ot)
    yb_ote, za_ote = otrecod(data, UnbalancedModality(iterations=5))
    ote = accuracy(data, yb_ote, za_ote)
    yb_sl, za_sl = otrecod(data, SimpleLearning())
    sl = accuracy(data, yb_sl, za_sl)
    
    println( " OT : $ot \t SL : $sl \t OTE : $ote ")

end
