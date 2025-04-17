using CSV
using DataFrames
using OptimalTransportDataIntegration


mB_values = ([0,0,0], [1,0,0], [1,1,0] , [1,2,0])

for mB = mB_values

    params = DataParameters(nA = 1000, nB = 1000, mB = mB)
        
    rng = DataGenerator(params)
    data = generate_data(rng)

        
    result_ot = otrecod(data, JointOTWithinBase())
    ot = accuracy(result_ot)
    result_ote = otrecod(data, JointOTBetweenBases(iterations=5))
    ote = accuracy(result_ote)
    result_sl = otrecod(data, SimpleLearning())
    sl = accuracy(result_sl)
    
    println( " OT : $ot \t SL : $sl \t OTE : $ote ")

end
