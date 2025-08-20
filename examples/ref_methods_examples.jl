using OptimalTransportDataIntegration

params = DataParameters()

rng = DataGenerator(params, discrete = false)

data = generate( rng ) 


result = otrecod( data, JointOTBetweenBasesRefJDOT())
println("JointOTBetweenBasesRefJDOT : $(accuracy(result))")

result = otrecod( data, JointOTBetweenBasesrefOTDAx())
println("JointOTBetweenBasesrefOTDAx : $(accuracy(result))")

result = otrecod( data, JointOTBetweenBasesrefOTDAyz())
println("JointOTBetweenBasesrefOTDAyz : $(accuracy(result))")

result = otrecod( data, JointOTBetweenBasesrefOTDAyz())
println("JointOTBetweenBasesrefOTDAyz : $(accuracy(result))")

result = otrecod( data, JointOTBetweenBases() ) 
println("JointOTBetweenBases : $(accuracy(result))")  
