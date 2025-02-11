"""
Unit-tests for the ProgramsMachine class.

Run as python test_programs_machine.py in console or via coverage
"""
from __future__ import annotations
import unittest

import numpy as np

from longitudinal_tomography.tracking import ProgramsMachine
from .. import commons

MACHINE_ARGS = commons.get_machine_args()
P_MACHINE_ARGS = MACHINE_ARGS.copy()
to_remove = ('vrf1', 'vrf2', 'vrf1dot', 'vrf2dot', 'b0', 'bdot', 'dturns',
             'phi12')
for elem in to_remove:
    P_MACHINE_ARGS.pop(elem)
P_MACHINE_ARGS['t_ref'] = 0.276


class TestProgramsMachine(unittest.TestCase):

    def test_values_at_turns_correct_length(self):
        machine = ProgramsMachine(8, voltage_program,
                                  momentum_program, [1, 2],
                                  phase_function=phase_program,
                                  **P_MACHINE_ARGS)

        self.assertEqual(1200, len(machine.time_at_turn),
                         msg='Wrong length of array: time_at_turn')
        self.assertEqual(1200, len(machine.omega_rev0),
                         msg='Wrong length of array: omega_rev0')
        self.assertEqual(1200, len(machine.phi0),
                         msg='Wrong length of array: phi0')
        self.assertEqual(1200, len(machine.drift_coef),
                         msg='Wrong length of array: drift_coef')
        self.assertEqual(1199, len(machine.deltaE0),
                         msg='Wrong length of array: deltaE0')
        self.assertEqual(1200, len(machine.beta0),
                         msg='Wrong length of array: beta0')
        self.assertEqual(1200, len(machine.eta0),
                         msg='Wrong length of array: eta0')
        self.assertEqual(1200, len(machine.e0),
                         msg='Wrong length of array: e0')
        self.assertEqual(1200, len(machine.vrf1_at_turn),
                         msg='Wrong length of array: vrf1_at_turn')
        self.assertEqual(1200, len(machine.vrf2_at_turn),
                         msg='Wrong length of array: vrf2_at_turn')

    def test_values_at_turns_correct_time_at_turn(self):
        args = P_MACHINE_ARGS.copy()
        args['nprofiles'] = 5
        args['machine_ref_frame'] = 2

        machine = ProgramsMachine(8, voltage_program,
                                  momentum_program, [1, 2],
                                  phase_function=phase_program,
                                  **args)

        correct = [-1.6129550637329442e-05, -1.5121453722496352e-05,
                   -1.4113356807663261e-05, -1.3105259892830171e-05,
                   -1.2097162977997081e-05, -1.1089066063163991e-05,
                   -1.0080969148330901e-05, -9.072872233497811e-06,
                   -8.064775318664721e-06, -7.056678403831631e-06,
                   -6.048581488998541e-06, -5.0404845741654505e-06,
                   -4.0323876593323604e-06, -3.0242907444992703e-06,
                   -2.0161938296661802e-06, -1.0080969148330901e-06, 0.0,
                   1.0080969148330901e-06, 2.0161938296661802e-06,
                   3.0242907444992703e-06, 4.0323876593323604e-06,
                   5.0404845741654505e-06, 6.048581488998541e-06,
                   7.056678403831631e-06, 8.064775318664721e-06,
                   9.072872233497811e-06, 1.0080969148330901e-05,
                   1.1089066063163991e-05, 1.2097162977997081e-05,
                   1.3105259892830171e-05, 1.4113356807663258e-05,
                   1.5121453722496352e-05]

        for tat, corr in zip(machine.time_at_turn, correct):
            self.assertAlmostEqual(tat, corr,
                                   msg='Error in calculation of time at turn')

    def test_values_at_turns_correct_omega_rev0(self):

        args = P_MACHINE_ARGS.copy()
        args['nprofiles'] = 5
        args['machine_ref_frame'] = 2

        machine = ProgramsMachine(8, voltage_program,
                                  momentum_program, [1, 2],
                                  phase_function=phase_program,
                                  **args)

        correct = [6232957.307404246, 6232958.474517781, 6232959.641630856,
                   6232960.80874347, 6232961.975855622, 6232963.142967314,
                   6232964.310078545, 6232965.477189314, 6232966.644299621,
                   6232967.811409467, 6232968.978518853, 6232970.145627777,
                   6232971.31273624, 6232972.479844241, 6232973.64695178,
                   6232974.814058861, 6232975.981165478, 6232977.148271635,
                   6232978.315377329, 6232979.482482564, 6232980.654209093,
                   6232982.074924732, 6232983.495639686, 6232984.916353958,
                   6232986.337067545, 6232987.75778045, 6232989.178492671,
                   6232990.599204209, 6232992.019915062, 6232993.4406252345,
                   6232994.861334722, 6232996.282043525, 6232997.702751645]

        for omega, corr in zip(machine.omega_rev0, correct):
            self.assertAlmostEqual(omega, corr,
                                   msg='Error in calculation of revolution '
                                       'frequency (omega_rev0)')

    def test_values_at_turns_correct_phi0(self):

        args = P_MACHINE_ARGS.copy()
        args['nprofiles'] = 5
        args['machine_ref_frame'] = 2

        machine = ProgramsMachine(8, voltage_program,
                                  momentum_program, [1, 2],
                                  phase_function=phase_program,
                                  **args)

        correct = [0.012957336821816726, 0.013049368308642274,
                   0.013141345162347044, 0.013233267594045545,
                   0.013325135834361042, 0.01341695005650427,
                   0.013508710467412493, 0.013600417277310765,
                   0.013692070682457855, 0.013783670878136199,
                   0.01387521808017292, 0.013966712473261858,
                   0.01405815422401691, 0.014149543596818545,
                   0.014240880726319278, 0.014332165823489859,
                   0.014423391923418417, 0.014493532859460555,
                   0.014563673649327485, 0.014599638207932027,
                   0.01279457142470893, 0.011023928774455771,
                   0.011094320842502645, 0.011164712554396935,
                   0.011235103977612583, 0.011305495095686319,
                   0.01137588590924272, 0.011446276427435557,
                   0.011516666650882102, 0.011587056558863818,
                   0.011657446194670937, 0.011727835520517012,
                   0.011798224547296144]

        for phi, corr in zip(machine.phi0, correct):
            self.assertAlmostEqual(phi, corr,
                                   msg='Error in calculation of synchronous '
                                       'phase at each turn (phi0)')

    def test_values_at_turns_correct_drift_coef(self):

        args = P_MACHINE_ARGS.copy()
        args['nprofiles'] = 5
        args['machine_ref_frame'] = 2

        machine = ProgramsMachine(8, voltage_program,
                                  momentum_program, [1, 2],
                                  phase_function=phase_program,
                                  **args)

        correct = [1.419505511043395e-08, 1.4195046668049025e-08,
                   1.4195038225672399e-08, 1.419502978330408e-08,
                   1.4195021340944071e-08, 1.4195012898592377e-08,
                   1.4195004456248972e-08, 1.4194996013913894e-08,
                   1.4194987571587122e-08, 1.4194979129268647e-08,
                   1.4194970686958483e-08, 1.4194962244656641e-08,
                   1.4194953802363092e-08, 1.4194945360077842e-08,
                   1.4194936917800919e-08, 1.419492847553228e-08,
                   1.4194920033271972e-08, 1.4194911591019962e-08,
                   1.4194903148776261e-08, 1.4194894706540863e-08,
                   1.4194886230882417e-08, 1.4194875954173356e-08,
                   1.4194865677476617e-08, 1.4194855400792184e-08,
                   1.4194845124120055e-08, 1.4194834847460243e-08,
                   1.4194824570812736e-08, 1.4194814294177535e-08,
                   1.4194804017554665e-08, 1.4194793740944075e-08,
                   1.4194783464345815e-08, 1.419477318775985e-08,
                   1.4194762911186208e-08]

        for drift, corr in zip(machine.drift_coef, correct):
            self.assertAlmostEqual(drift, corr,
                                   msg='Error in calculation of drift '
                                       'coefficient (drift_coef)')

    def test_values_at_turns_correct_deltaE0(self):

        args = P_MACHINE_ARGS.copy()
        args['nprofiles'] = 5
        args['machine_ref_frame'] = 2

        machine = ProgramsMachine(8, voltage_program,
                                  momentum_program, [1, 2],
                                  phase_function=phase_program,
                                  **args)

        correct = [76.12663292884827, 76.12663292884827, 76.12663269042969,
                   76.12663292884827, 76.12663269042969, 76.12663292884827,
                   76.12663292884827, 76.12663269042969, 76.12663292884827,
                   76.12663292884827, 76.12663269042969, 76.12663292884827,
                   76.12663292884827, 76.12663269042969, 76.12663269042969,
                   76.12663292884827, 76.12663292884827, 76.12663269042969,
                   76.12663292884827, 76.42809557914734, 92.66892576217651,
                   92.66892552375793, 92.66892552375793, 92.66892552375793,
                   92.66892552375793, 92.66892552375793, 92.66892576217651,
                   92.66892552375793, 92.66892552375793, 92.66892552375793,
                   92.66892552375793, 92.66892552375793]

        for dE0, corr in zip(machine.deltaE0, correct):
            self.assertAlmostEqual(dE0, corr,
                                   msg='Error in calculation of energy '
                                       'difference of synch part pr turn '
                                       '(deltaE0)')

    def test_values_at_turns_correct_beta0(self):

        args = P_MACHINE_ARGS.copy()
        args['nprofiles'] = 5
        args['machine_ref_frame'] = 2

        machine = ProgramsMachine(8, voltage_program,
                                  momentum_program, [1, 2],
                                  phase_function=phase_program,
                                  **args)

        correct = [0.5197726911632519, 0.5197727884900445, 0.5197728858167987,
                   0.5197729831435145, 0.5197730804701918, 0.5197731777968306,
                   0.5197732751234311, 0.5197733724499929, 0.5197734697765163,
                   0.5197735671030013, 0.5197736644294478, 0.5197737617558558,
                   0.5197738590822254, 0.5197739564085566, 0.5197740537348492,
                   0.5197741510611036, 0.5197742483873192, 0.5197743457134965,
                   0.5197744430396353, 0.5197745403657357, 0.5197746380772105,
                   0.5197747565521421, 0.5197748750270167, 0.5197749935018343,
                   0.519775111976595, 0.5197752304512986, 0.5197753489259452,
                   0.5197754674005349, 0.5197755858750674, 0.5197757043495432,
                   0.5197758228239618, 0.5197759412983236, 0.5197760597726283]

        for beta, corr in zip(machine.beta0, correct):
            self.assertAlmostEqual(beta, corr,
                                   msg='Error in calculation of relativistic '
                                       'beta (beta0)')

    def test_values_at_turns_correct_eta0(self):

        args = P_MACHINE_ARGS.copy()
        args['nprofiles'] = 5
        args['machine_ref_frame'] = 2

        machine = ProgramsMachine(8, voltage_program,
                                  momentum_program, [1, 2],
                                  phase_function=phase_program,
                                  **args)

        correct = [0.6703479497588644, 0.670347848583237, 0.6703477474076307,
                   0.6703476462320453, 0.6703475450564811, 0.6703474438809379,
                   0.6703473427054155, 0.6703472415299145, 0.6703471403544344,
                   0.6703470391789753, 0.6703469380035373, 0.6703468368281205,
                   0.6703467356527245, 0.6703466344773494, 0.6703465333019957,
                   0.6703464321266627, 0.670346330951351, 0.6703462297760602,
                   0.6703461286007905, 0.6703460274255418, 0.6703459258496584,
                   0.6703458026891149, 0.6703456795286028, 0.6703455563681217,
                   0.6703454332076717, 0.6703453100472531, 0.6703451868868654,
                   0.670345063726509, 0.670344940566184, 0.6703448174058898,
                   0.6703446942456269, 0.6703445710853951, 0.6703444479251947]

        for eta, corr in zip(machine.eta0, correct):
            self.assertAlmostEqual(eta, corr,
                                   msg='Error in calculation of phase slip '
                                       'factor (eta0)')

    # This array is tested as integers due to its high values.
    def test_values_at_turns_correct_e0(self):

        args = P_MACHINE_ARGS.copy()
        args['nprofiles'] = 5
        args['machine_ref_frame'] = 2

        machine = ProgramsMachine(8, voltage_program,
                                  momentum_program, [1, 2],
                                  phase_function=phase_program,
                                  **args)

        correct = [1098287788.7333486, 1098287864.8599815, 1098287940.9866145,
                   1098288017.1132472, 1098288093.23988, 1098288169.3665128,
                   1098288245.4931457, 1098288321.6197786, 1098288397.7464113,
                   1098288473.8730443, 1098288549.9996772, 1098288626.1263099,
                   1098288702.2529428, 1098288778.3795757, 1098288854.5062084,
                   1098288930.632841, 1098289006.759474, 1098289082.886107,
                   1098289159.0127397, 1098289235.1393726, 1098289311.5674682,
                   1098289404.236394, 1098289496.9053195, 1098289589.574245,
                   1098289682.2431705, 1098289774.912096, 1098289867.5810215,
                   1098289960.2499473, 1098290052.9188728, 1098290145.5877984,
                   1098290238.2567239, 1098290330.9256494, 1098290423.594575]

        for e0, corr in zip(machine.e0, correct):
            self.assertEqual(e0, corr,
                             msg='Error in calculation of energy '
                                 'of synch. particle (e0)')


# c275 to c278 for the ISOLDE cycle in the PSB
voltage_program = np.array(
    [[0.275, 0.27510080967400036, 0.27520161934799775, 0.27530242902199514,
      0.27540323869599254, 0.27550404836998993, 0.27560485804386414,
      0.2760000262064593, 0.2770008413556847, 0.2780009830187509],
     [4525.002016339983, 4525.13803754534, 4525.04871913365, 4524.65694972447,
      4523.303440638951, 4518.739264351981, 4592.280742100307,
      8818.071583339146, 9401.939719063896, 9860.669653756278],
     [3393.751512254988, 3393.8535281590057, 3393.7865393502375,
      3393.4927122933527, 3392.4775804792134, 3389.054448263986,
      3444.210556575231, 6613.553687504361, 7051.454789297923,
      7395.50224031721]]
)
phase_program = np.array(
    [[0.275, 0.27510080967400036, 0.27520161934799775, 0.27530242902199514,
      0.27540323869599254, 0.27550404836998993, 0.27560485804386414,
      0.2760000262064593, 0.2770008413556847, 0.2780009830187509],
     [3.141592653589793, 3.141592653589793, 3.141592653589793,
      3.141592653589793, 3.141592653589793, 3.141592653589793,
      3.141592653589793, 3.141592653589793, 3.141592653589793,
      3.141592653589793],
     [3.141576227067777, 3.1415985586782416, 3.141579919997621,
      3.1416376866722406, 3.14142520589179, 3.1422180288908788,
      3.1387891772057266, 3.12046578386211, 3.074796197055706,
      3.0382161961689236]]
)
momentum_program = np.array(
    [[0.275, 0.27500606060606064, 0.2750121212121212, 0.27501818181818183,
      0.27502424242424245, 0.27503030303030307, 0.27503636363636363,
      0.27504242424242425, 0.2750484848484849, 0.2750545454545455,
      0.27506060606060606, 0.2750666666666667, 0.2750727272727273,
      0.2750787878787879, 0.2750848484848485, 0.2750909090909091,
      0.27509696969696973, 0.27510303030303035, 0.2751090909090909,
      0.27511515151515153, 0.27512121212121216, 0.2751272727272727,
      0.27513333333333334, 0.27513939393939396, 0.2751454545454546,
      0.27515151515151515, 0.27515757575757577, 0.2751636363636364,
      0.275169696969697, 0.2751757575757576, 0.2751818181818182,
      0.2751878787878788, 0.27519393939393944, 0.2752, 0.2752060606060606,
      0.27521212121212124, 0.27521818181818186, 0.2752242424242424,
      0.27523030303030305, 0.27523636363636367, 0.27524242424242423,
      0.27524848484848485, 0.2752545454545455, 0.2752606060606061,
      0.27526666666666666, 0.2752727272727273, 0.2752787878787879,
      0.2752848484848485, 0.2752909090909091, 0.2752969696969697,
      0.2753030303030303, 0.27530909090909095, 0.2753151515151515,
      0.27532121212121213, 0.27532727272727275, 0.2753333333333334,
      0.27533939393939394, 0.27534545454545456, 0.2753515151515152,
      0.2753575757575758, 0.27536363636363637, 0.275369696969697,
      0.2753757575757576, 0.27538181818181817, 0.2753878787878788,
      0.2753939393939394, 0.27540000000000003, 0.2754060606060606,
      0.2754121212121212, 0.27541818181818184, 0.27542424242424246,
      0.275430303030303, 0.27543636363636365, 0.27544242424242427,
      0.2754484848484849, 0.27545454545454545, 0.2754606060606061,
      0.2754666666666667, 0.2754727272727273, 0.2754787878787879,
      0.2754848484848485, 0.2754909090909091, 0.2754969696969697,
      0.2755030303030303, 0.2755090909090909, 0.27551515151515155,
      0.2755212121212121, 0.27552727272727273, 0.27553333333333335,
      0.275539393939394, 0.27554545454545454, 0.27555151515151516,
      0.2755575757575758, 0.2755636363636364, 0.27556969696969696,
      0.2755757575757576, 0.2755818181818182, 0.2755878787878788,
      0.2755939393939394, 0.2756, 0.2756, 0.2756808080808081,
      0.27576161616161615, 0.2758424242424243, 0.27592323232323235,
      0.2760040404040404, 0.2760848484848485, 0.27616565656565656,
      0.2762464646464647, 0.27632727272727275, 0.2764080808080808,
      0.2764888888888889, 0.27656969696969697, 0.27665050505050504,
      0.27673131313131316, 0.27681212121212123, 0.2768929292929293,
      0.27697373737373737, 0.27705454545454544, 0.27713535353535357,
      0.27721616161616164, 0.2772969696969697, 0.2773777777777778,
      0.27745858585858585, 0.277539393939394, 0.27762020202020204,
      0.2777010101010101, 0.2777818181818182, 0.27786262626262626,
      0.2779434343434344, 0.27802424242424245, 0.2781050505050505,
      0.2781858585858586, 0.27826666666666666, 0.2783474747474748,
      0.27842828282828286, 0.27850909090909093, 0.278589898989899,
      0.27867070707070707, 0.27875151515151514, 0.27883232323232326,
      0.27891313131313133, 0.2789939393939394, 0.2790747474747475,
      0.27915555555555555, 0.27923636363636367, 0.27931717171717174,
      0.2793979797979798, 0.2794787878787879, 0.27955959595959595,
      0.2796404040404041, 0.27972121212121215, 0.2798020202020202,
      0.2798828282828283, 0.27996363636363636, 0.2800444444444445],
     [570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830158.7660657, 570830158.7660657,
      570830158.7660657, 570830157.0537509, 570831481.8766614,
      570835442.2872213, 570842017.1984887, 570851185.5234518,
      570862926.1749934, 570877218.0660315, 570894040.1096944,
      570913371.2187595, 570935190.3063556, 570959476.2852249,
      570986208.0686013, 571015364.5691569, 571046924.7000552,
      571080867.3741794, 571117171.5044819, 571155816.0039864,
      571196779.7856456, 571240041.7623425, 571285580.8471352,
      571333375.9528363, 571383405.9925042, 571435649.879092,
      571490086.5255876, 571546694.8449439, 571605453.7501141,
      571666342.154016, 571729338.9695675, 571794423.1098621,
      571861573.4876771, 571930769.0162114, 572001988.6083125,
      572075211.1768633, 572150415.6349572, 572227580.8954067,
      572306685.8713756, 572387709.4755359, 572470630.6211213,
      572555428.2209446, 572642081.1880289, 572730568.4352922,
      572820868.8756523, 572912961.4222376, 573006824.9877557,
      573102438.485405, 573199780.8280332, 573298830.9285582,
      573399567.7001088, 573501970.0553217, 573606016.9075712,
      573711687.1695296, 573818959.7542201, 573927813.5746657,
      574038227.5437495, 574150180.5745294, 574263651.5798182]]
)
