using DelimitedFiles
using OptimalTransportDataIntegration

function main(nsimulations::Int)

    outfile = "covariates_shift_assumption.csv"
    mB_values = ([0,0,0], [1,0,0], [1,1,0] , [1,2,0])
    reg_m_values = [0.01 0.05 0.1 0.25 0.5 0.75 1]
    header = ["id", "mB", "reg_m", "estimation"]

    open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for i = 1:nsimulations
            for (j,mB) in enumerate(mb_values)
                params = DataParameters(mB = mB)
                data = generate_xcat_ycat(params)

                for reg_m in reg_m_values

                     est = otrecod(data, UnbalancedModality(reg = 0.01, reg_m = reg_m))
                     writedlm(io, [i j reg_m est ])

                 end

        end

    end

end

nsimulations = 100

@time main(nsimulations)
