using OptimalTransportDataIntegration

#params = DataParameters(mB = [5, 5, 5], aA = [1.,1,1], aB = [1.,1,1])
params = DataParameters(mA = [0],mB = [5], covA=[1], aA = [1.], aB = [1.])
rng = DataGenerator(params, scenario = 2, discrete = false)

data = generate( rng ) 


println(data)

# result = otrecod( data, JointOTBetweenBasesRefJDOT())
# println("JointOTBetweenBasesRefJDOT : $(accuracy(result))")

# result = otrecod( data, JointOTBetweenBasesrefOTDAx())
# println("JointOTBetweenBasesrefOTDAx : $(accuracy(result))")
# 
# result = otrecod( data, JointOTBetweenBasesrefOTDAyz())
# println("JointOTBetweenBasesrefOTDAyz : $(accuracy(result))")
# 
# result = otrecod( data, JointOTBetweenBasesrefOTDAyz())
# println("JointOTBetweenBasesrefOTDAyz : $(accuracy(result))")
# 
# result = otrecod( data, JointOTBetweenBases() ) 
# println("JointOTBetweenBases : $(accuracy(result))")  
