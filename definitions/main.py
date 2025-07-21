from encoder_only.intrinsic_bias.similarity_based.weat import main as weat
from encoder_only.intrinsic_bias.similarity_based.seat import main as seat
from encoder_only.intrinsic_bias.similarity_based.ceat import main as ceat

from encoder_only.intrinsic_bias.probability_based.masked_token_metrics.lbps import main as lbps
from encoder_only.intrinsic_bias.probability_based.masked_token_metrics.cbs import main as cbs
from encoder_only.intrinsic_bias.probability_based.masked_token_metrics.disco import main as disco

from encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.ppl import main as ppl
from encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.aul import main as aul
from encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.aula import main as aula
from encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cat import main as cat
from encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.icat import main as icat
from encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps import main as cps

from encoder_only.extrinsic_bias.equal_opportunity import main as gap
from encoder_only.extrinsic_bias.fair_inference import main as nn
from encoder_only.extrinsic_bias.fair_inference import main as fn
from encoder_only.extrinsic_bias.fair_inference import main as t_0.5
from encoder_only.extrinsic_bias.fair_inference import main as t_0.7
from encoder_only.extrinsic_bias.context_based import main as s_amb
from encoder_only.extrinsic_bias.fair_inference import main as s_dis

from decoder_only.intrinsic_bias.attention_head_based.nie import main as nie
from decoder_only.intrinsic_bias.attention_head_based.gbe import main as gbe

from decoder_only.intrinsic_bias.stereotype_association.sll import main as sll
from decoder_only.intrinsic_bias.stereotype_association.ca import main as ca

from decoder_only.extrinsic_bias.counterfactual_fairness.cr import main as cr
from decoder_only.extrinsic_bias.counterfactual_fairness.ctf import main as ctf

from decoder_only.extrinsic_bias.performance_disparity.ad import main as ad
from decoder_only.extrinsic_bias.performance_disparity.ba import main as ba
from decoder_only.extrinsic_bias.performance_disparity.sns import main as sns

from decoder_only.extrinsic_bias.demographic_representation.drd import main as drd
from decoder_only.extrinsic_bias.demographic_representation.dnp import main as dnp

from encoder_decoder.intrinsic_bias.algorithmic_disparity.lfp import main as lfp
from encoder_decoder.intrinsic_bias.algorithmic_disparity.mcd import main as mcd

from encoder_decoder.intrinsic_bias.stereotype_association.sd import main as sd
from encoder_decoder.intrinsic_bias.stereotype_association.sva import main as sva

from encoder_decoder.extrinsic_bias.position_based_disparity.npd import main as npd

from encoder_decoder.extrinsic_bias.fair_inference.ibs import main as ibs

from encoder_decoder.extrinsic_bias.individual_fairness.ss import main as ss

from encoder_decoder.extrinsic_bias.counterfactual_fairness.auc import main as auc




import sys

def print_invalid_command():
    print("Invalid command!")

if sys.argv[1] == "medium":
    if sys.argv[2] == "intrinsic":
        if sys.argv[3] == "similarity":
            if sys.argv[4] == "weat":
                weat.run_experiment()
            elif sys.argv[4] == "seat":
                seat.run_experiment()
            elif sys.argv[4] == "ceat":
                ceat.run_experiment()
            else: 
                print_invalid_command()

        elif sys.argv[3] == "probability":
            if sys.argv[4] == "disco":
                disco.run_experiment()
            elif sys.argv[4] == "lbps":
                lbps.run_experiment()
            elif sys.argv[4] == "cbs":
                cbs.run_experiment()
            elif sys.argv[4] == "ppl":
                ppl.run_experiment()
            elif sys.argv[4] == "cps":
                cps.run_experiment()
            elif sys.argv[4] == "cat":
                cat.run_experiment()
            elif sys.argv[4] == "icat":
                icat.run_experiment()
            elif sys.argv[4] == "aul":
                aul.run_experiment()
            elif sys.argv[4] == "aula":
                aula.run_experiment()
            else: 
                print_invalid_command()

        else: 
            print_invalid_command()

    elif sys.argv[2] == "extrinsic":
        if sys.argv[3] == "classification":
            classification.run_experiment()
        if sys.argv[3] == "qa":
            qa.run_experiment()
    
    else:
        print_invalid_command()

elif sys.argv[1] == "large":
    if sys.argv[2] == "dr":
        if sys.argv[3] == "exp1":
            dr_exp1.run_experiment()
        elif sys.argv[3] == "exp2":
            dr_exp2.run_experiment()
        elif sys.argv[3] == "exp3":
            dr_exp3.run_experiment()
        else: 
            print_invalid_command()

    if sys.argv[2] == "sa":
        if sys.argv[3] == "exp1":
            sa_exp1.run_experiment()
        elif sys.argv[3] == "exp2":
            sa_exp2.run_experiment()
        elif sys.argv[3] == "exp3":
            sa_exp3.run_experiment()
        else: 
            print_invalid_command()

    if sys.argv[2] == "cf":
        if sys.argv[3] == "exp1":
            cf_exp1.run_experiment()
        elif sys.argv[3] == "exp2":
            cf_exp2.run_experiment()
        else: 
            print_invalid_command()
        
    if sys.argv[2] == "pd":
        if sys.argv[3] == "exp1":
            pd_exp1.run_experiment()
        elif sys.argv[3] == "exp2":
            pd_exp2.run_experiment()
        elif sys.argv[3] == "exp3":
            pd_exp3.run_experiment()
        else: 
            print_invalid_command()

else :
    print_invalid_command()
