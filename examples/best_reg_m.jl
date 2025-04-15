using DelimitedFiles
using OptimalTransportDataIntegration

function main(nsimulations::Int)

    outfile = "best_reg_m_values_with_different_mb.csv"
    mb_values = [[1,0,0], [1,1,0], [1,2,0], [1,2,1]]
    reg_m_values = [0.01 0.05 0.1 0.25 0.5 0.75 1]
    header = ["id", "mB", "reg_m1", "reg_m2", "estyb", "estza", "accuracy"]

    open(outfile, "w") do io

        writedlm(io, hcat(header...))

        for i = 1:nsimulations

            for mB in mb_values

                @show mB
                params = DataParameters(mB = mB)
                data = generate_xcat_ycat(params)

                for reg_m1 in reg_m_values, reg_m2 in reg_m_values

                      yb, za  = otrecod(data, JointOTBetweenBases(reg = 0.001, reg_m1 = reg_m1, reg_m2 = reg_m2))
                      estyb, estza, est = accuracy( data, yb, za )
                      writedlm(io, [i repr(mB) reg_m1 reg_m2 estyb estza est ])

                end

            end

        end

    end

end

nsimulations = 1000

@time main(nsimulations)
