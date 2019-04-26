#!/share/home/zyzhu/anaconda3/bin/python

"""
Non-Consistent GGA program

Usage:
    ncgga.py -h
    ncgga.py [--no-exec] [--confirm] [-q <queue>] [-p <cpu-core>] [-n <node>] [-m <memory>] <gaussian-input>

Options:
    -h --help                   Show this screen.
    --no-exec                   Do not submit bsub queue or execute.
    --confirm                   Do numerical derivative, and show maximum deviation of numerical and
                                analytical derivative
    -q --queue <queue>          Bsub queue name, e.g. `xp24mc2`, `single`.
    -p --process <cpu-core>     CPU threading numbers.
                                    Default value is the default largest threading numbers of a node,
                                    or `1` for single, `8` for small.
                                    Note that this option can be OVERRIDEN BY gaussian-input %nproc value.
    -m --mem <memory>           Memory in GB in a node.
                                    Default value is the default largest memory of a node * 0.8.
                                    or `10` for single, `10` for small.
                                    Note that this option can be OVERRIDEN BY gaussian-input %mem value.
    -n --node <node>            Specify node name, eg. `xc07n10`.
    <gaussian-input>            File name usually have suffix .gjf or .com.
                                    DO NOT use .py or .bsub, program does not check suffix.
"""
from docopt import docopt
import os
import typing
import re
import stat
import warnings


def parse_gaussian_input(lines: typing.List[str], config_dict: dict):
    line_cur_num = 0

    # special settings
    for line_num in range(line_cur_num, len(lines)):
        line = lines[line_num].strip().lower()
        if re.match("^%(.*)$", line):
            try:
                line_rg = re.match(r"^%([a-z]+?)( *?)=( *?)([0-9]+?)([a-z]*?)$", line)
                line_group = line_rg.groups()
                if line_group[0] == "mem":
                    mem = float(line_group[3])
                    mem_prefix = {
                        "kb": 1 / (1024 * 1024),
                        "mb": 1 / 1024,
                        "gb": 1,
                        "tb": 1024,
                        "kw": 1 / (1024 * 1024 / 8),
                        "mw": 1 / (1024 / 8),
                        "gw": 8,
                        "tw": 1024 * 8
                    }
                    mem *= mem_prefix[line_group[4]]
                    if config_dict["queue"] is not None:
                        warnings.warn("Overwrite default memory {} GB to {} GB.\n"
                                      .format(config_dict["mem"], mem))
                    config_dict["mem"] = mem
                    if mem < 8:
                        raise ValueError("Memory should not less than 8GB!")
                elif line_group[0] == "nproc" or line_group[0] == "nprocshared":
                    if line_group[4] != "":
                        raise AttributeError("Process numbers should not have any suffix like `"
                                             + line_group[4] + "`\n")
                    cpu = int(line_group[3])
                    if config_dict["queue"] is not None:
                        warnings.warn("Overwrite default cpu thread number {} to {}.\n"
                                      .format(config_dict["cpu"], cpu))
                    config_dict["cpu"] = cpu
            except AttributeError:
                raise ValueError("Cannot parse line: `" + line + "`\n").with_traceback(...)
        elif re.match("^#(.*)$", line):
            words = line.split()
            for word in words:
                if word[0] == "#":
                    # #p, #n, #t, #
                    if len(word) > 2:
                        raise ValueError("No such print level! `" + word + "`\n")
                    if len(word) == 0 or word[1] == "n":
                        config_dict["loglevel"] = "n"
                    elif word[1] == "t":
                        config_dict["loglevel"] = "t"
                    elif word[1] == "p":
                        config_dict["loglevel"] = "p"
                    else:
                        raise AttributeError("No such print level! `" + word + "`\n")
                elif word == "force":
                    # force
                    config_dict["job"] = "force"
                elif word == "freq":
                    # freq
                    config_dict["job"] = "freq"
                elif re.match(r"^int(egral)?\(grid( *?)=( *?)([0-9]+?)\)$", word):
                    # Int(Grid=99590)
                    grid = int(re.match(r"^int(egral)?\(grid( *?)=( *?)([0-9]+?)\)$", word).group(4))
                    assert grid > 1000
                    config_dict["grid_rad"] = int(grid / 1000)
                    config_dict["grid_sph"] = grid % 1000
                elif re.match(r"^(\w+?)-(\w+?)/([\w-]+?)$", word):
                    # B3LYP-PBE0/aug-cc-pVTZ
                    xc_group = re.match(r"^(\w+?)-(\w+?)/([\w-]+?)$", word).groups()
                    config_dict["scf_xc"] = xc_group[0]
                    config_dict["nc_xc"] = xc_group[1]
                    config_dict["basis"] = xc_group[2]
                elif word == "6d" or word == "10f":
                    config_dict["cart"] = True
                elif word == "5d" or word == "7f":
                    config_dict["cart"] = False
                elif word == "nosymm":
                    pass
                elif word == "opt":
                    config_dict["opt_level"] = "tight"
                elif re.match(r"^opt\((\w+?)\)$", word):
                    config_dict["opt_level"] = re.match(r"^opt\((\w+?)\)$", word).group(1)
                else:
                    raise ValueError("Cannot parse word `" + word + "`\n")
        else:
            line_cur_num = line_num
            break

    # Next four lines
    assert lines[line_cur_num].strip() == ""
    assert lines[line_cur_num + 2].strip() == ""
    assert lines[line_cur_num + 3].split() == ["0", "1"]
    config_dict["title"] = lines[line_cur_num + 1].strip()
    line_cur_num += 4
    for line_num in range(line_cur_num, len(lines)):
        line = lines[line_num].strip()
        if line != "":
            config_dict["mol"].append(line)
            if line_num == len(lines) - 1:  # Situation that no finalize after mol definition
                line_cur_num = line_num
                break
        else:
            line_cur_num = line_num
            break

    # Finalize: self-defined xc functional
    line_cur_num += 1
    for line_num in range(line_cur_num, len(lines)):
        line = lines[line_num].strip().lower()
        if line != "":
            xc_group = re.match(r"^(\w+?)( *?)=( *?)(.+?)$", line)
            if xc_group is None:
                raise ValueError("Cannot parse line `" + line + "`\n")
            if xc_group.group(1) == config_dict["scf_xc"]:
                config_dict["scf_xc"] = xc_group.group(4)
            elif xc_group.group(1) == config_dict["nc_xc"]:
                config_dict["nc_xc"] = xc_group.group(4)
            else:
                raise ValueError("No such exchange-correlation functional: `" + xc_group.group(1) + "`\n")

    # B3LYP specilize
    if config_dict["scf_xc"] == "b3lyp":
        warnings.warn("B3LYP change to B3LYPG refer to Gaussian type of B3LYP.\n"
                      "For gamess type, change B3LYP to B3LYPVWN3.")
        config_dict["scf_xc"] = "b3lypg"
    elif config_dict["scf_xc"] == "b3lypvwn3":
        config_dict["scf_xc"] = "b3lyp"
    if config_dict["nc_xc"] == "b3lyp":
        warnings.warn("B3LYP change to B3LYPG refer to Gaussian type of B3LYP.\n"
                      "For gamess type, change B3LYP to B3LYPVWN3.")
        config_dict["nc_xc"] = "b3lypg"
    elif config_dict["nc_xc"] == "b3lypvwn3":
        config_dict["nc_xc"] = "b3lyp"

    return config_dict


def test_config(config_dict: dict):

    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = "\n".join(config_dict["mol"])
    mol.basis = config_dict["basis"]
    mol.cart = config_dict["cart"]
    mol.build()

    grids = dft.Grids(mol)
    grids.becke_scheme = dft.grid.stratmann
    grids.atom_grid = (config_dict["grid_rad"], config_dict["grid_sph"])
    grids.build()

    assert dft.libxc.xc_type(config_dict["scf_xc"]) == "GGA" or dft.libxc.xc_type(config_dict["scf_xc"]) == "HF"
    assert dft.libxc.xc_type(config_dict["nc_xc"]) == "GGA"

    return


def write_config(config_dict: dict):

    loglevel = 2
    if config_dict["loglevel"] in ["t"]:
        loglevel = 0
    molverbose = 0
    if config_dict["loglevel"] in ["p"]:
        molverbose = 4
    config_list = [
        "#!/share/home/zyzhu/anaconda3/bin/python",
        "import sys",
        "sys.path.insert(0, '/share/home/zyzhu/Documents-Shared/HF_DFT_related/src')",
        "",
        "import os",
        "MAXCORE = '{}'".format(str(config_dict["cpu"])),
        "MAXMEM = '{}'".format(str((config_dict["mem"] - 2) * 0.6)),
        "MAXMOLMEM = '{}'".format(str((config_dict["mem"] - 2) * 0.3)),
        "os.environ['MAXMEM'] = MAXMEM",
        "os.environ['OMP_NUM_THREADS'] = MAXCORE",
        "os.environ['OPENBLAS_NUM_THREADS'] = MAXCORE",
        "os.environ['MKL_NUM_THREADS'] = MAXCORE",
        "os.environ['VECLIB_MAXIMUM_THREADS'] = MAXCORE",
        "os.environ['NUMEXPR_NUM_THREADS'] = MAXCORE",
        "os.environ['LOGLEVEL'] = '{}'".format(loglevel),
        "",
        "from hf_helper import HFHelper",
        "from gga_helper import GGAHelper",
        "from ncgga_engine import NCGGAEngine",
        "from utilities import timing_level",
        "from numeric_helper import NumericDiff",
        "from optimize_helper import OptimizeHelper",
        "from pyscf import gto, dft, lib",
        "\n",
        "print('Job name: {}')".format(config_dict["title"]),
        "",
        "mol = gto.Mole()",
        "mol.atom = '''\n{}\n'''".format("\n".join(config_dict["mol"])),
        "mol.basis = '{}'".format(config_dict["basis"]),
        "mol.cart = {}".format(config_dict["cart"]),
        "mol.verbose = {}".format(molverbose),
        "mol.max_memory = int(float(MAXMOLMEM) * 1024)",
        "mol.build()",
        "",
        "def mol_to_grids(mol):",
        "    grids = dft.Grids(mol)",
        "    grids.becke_scheme = dft.grid.stratmann",
        "    grids.atom_grid = ({}, {})".format(config_dict["grid_rad"], config_dict["grid_sph"]),
        "    grids.build()",
        "    return grids",
        "\n",
    ]

    helper_list = [
        "",
        "grids = mol_to_grids(mol)",
        "scfh = GGAHelper(mol, '{}', grids)".format(config_dict["scf_xc"]),
        "nch = GGAHelper(mol, '{}', grids, init_scf=False)".format(config_dict["nc_xc"]),
        "ncgga = NCGGAEngine(scfh, nch)",
    ]
    if config_dict["scf_xc"] == "hf":
        helper_list[0] = "hfh = HFHelper(mol)"

    e_list = [
        "@timing_level(0)",
        "def get_E_0():",
        "    E_0 = ncgga.E_0",
        "    print('E_0: {:18.10E}'.format(E_0))",
        "    return E_0",
        "\n",
        "@timing_level(0)",
        "def get_E_1():",
        "    E_1 = ncgga.E_1",
        "    print('E_1:')\n"
        "    for array in E_1:\n"
        "        print(''.join(['{:18.10E}'] * 3).format(*array))",
        "    return E_1",
        "\n",
        "@timing_level(0)",
        "def get_E_2():",
        "    E_2 = ncgga.E_2",
        "    for (A, arrayA) in enumerate(E_2):\n"
        "        for (B, arrayAB) in enumerate(arrayA):\n"
        "            print(A, ', ', B)\n"
        "            for array in arrayAB:\n"
        "                print(''.join(['{:18.10E}'] * 3).format(*array))",
        "    return E_2",
        "\n",
        "@timing_level(0)",
        "def mol_to_E_0(mol):",
        "\n    ".join(helper_list),
        "    return ncgga.E_0",
        "\n",
        "@timing_level(0)",
        "def mol_to_E_1(mol):",
        "\n    ".join(helper_list),
        "    return ncgga.E_1",
        "\n",
        "@timing_level(0)",
        "def confirm_E_1(E_1):",
        "    num_E_1 = NumericDiff(mol, mol_to_E_0).get_numdif()",
        "    print('Maximum difference of E_1: {:20.10E}'.format((num_E_1 - E_1).max()))",
        "    print('Minimum difference of E_1: {:20.10E}'.format((num_E_1 - E_1).min()))",
        "\n",
        "@timing_level(0)",
        "def confirm_E_2(E_2):",
        "    num_E_2 = NumericDiff(mol, mol_to_E_1, deriv=2).get_numdif()",
        "    print('Maximum difference of E_2: {:20.10E}'.format((num_E_2 - E_2).max()))",
        "    print('Minimum difference of E_2: {:20.10E}'.format((num_E_2 - E_2).min()))",
        "\n",
        "@timing_level(0)",
        "def mol_to_E_0_E_1(mol):",
        "\n    ".join(helper_list),
        "    E_0, E_1 = ncgga.E_0, ncgga.E_1",
        "    print('In mol_to_E_0_E_1: ')",
        "    print('E_0: {}'.format(E_0))",
        "    print('Molecule Geometry:')",
        "    print(mol.atom_coords() * lib.param.BOHR)",
        "    return E_0, E_1",
        "\n",
        "E_0 = get_E_0()",
        "E_1 = get_E_1()",
        "E_2 = get_E_2()",      # -7
        "confirm_E_1(E_1)",     # -6
        "confirm_E_2(E_2)",     # -5
        "mol_optimized = OptimizeHelper(mol).optimize(mol_to_E_0_E_1)",     # -4
        "print('Optimized geometry:')",                                     # -3
        "print(mol_optimized.atom_coords() * lib.param.BOHR)",              # -2
        "\n",                   # -1
    ]
    if config_dict["job"] == "force":
        e_list[-7] = e_list[-5] = ""
    if not config_dict["confirm"]:
        e_list[-6] = e_list[-5] = ""
    if config_dict["opt_level"] is None:
        e_list[-4] = e_list[-3] = e_list[-2] = ""
    if config_dict["opt_level"] == "verytight":
        e_list[-4] = "mol_optimized = OptimizeHelper(mol).verytight().optimize(mol_to_E_0_E_1)"

    prefix = config_dict["prefix"]
    with open(prefix + ".py", "w") as fwrite:
        fwrite.writelines("\n".join(config_list + helper_list + e_list))

    os.chmod(prefix + ".py", stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    return


def exec_program(config_dict: dict):
    prefix = config_dict["prefix"]
    if config_dict["queue"] is None:
        os.system("/share/home/zyzhu/anaconda3/bin/python " + prefix + ".py")
    else:
        os.system("bsub < " + prefix + ".bsub")


def prepare_bsub(config_dict: dict):
    prefix = config_dict["prefix"]
    bsub_list = [
        "#BSUB -n {}".format(config_dict["cpu"]),
        "#BSUB -q {}".format(config_dict["queue"]),
        "#BSUB -J {}".format(config_dict["prefix"]),
        "#BSUB -o {}-%J.out".format(config_dict["prefix"]),
        "",
        "",
        "export PYSCF_TMPDIR=/scratch/tmp",
        "./{}.py > {}.log 2>&1".format(config_dict["prefix"], config_dict["prefix"]),
        "",
    ]
    if config_dict["node"] is not None:
        bsub_list[4] = "#BSUB -m {}".format(config_dict["node"])

    with open(prefix + ".bsub", "w") as fwrite:
        fwrite.writelines("\n".join(bsub_list))


if __name__ == '__main__':

    # Parse arguments
    arguments = docopt(__doc__)
    print(arguments)

    # Parse input file
    assert os.path.exists(arguments["<gaussian-input>"])
    file_name = os.path.basename(arguments["<gaussian-input>"])
    dir_name = os.path.dirname(arguments["<gaussian-input>"])
    if dir_name == "":
        dir_name = "."
    file_prefix, file_suffix = file_name.rsplit('.', 1)
    os.chdir(dir_name)  # Entering to file directory to working directory
    with open(file_name, "r") as fread:
        file_lines = fread.readlines()

    # Construct conf dictionary
    conf = {
        "prefix": file_prefix,
        "mem": 10,
        "cpu": 1,
        "cart": False,
        "loglevel": "n",
        "scf_xc": None,
        "nc_xc": None,
        "basis": None,
        "grid_rad": 75,
        "grid_sph": 302,
        "opt_level": None,
        "job": "freq",
        "title": "",
        "mol": [],
        "queue": arguments["--queue"],
        "node": arguments["--node"],
        "confirm": arguments["--confirm"],
    }

    # Prepare bsub file if needed
    if arguments["--queue"] is not None:
        if re.match("^xp([0-9]+?)mc([0-9]+?)$", arguments["--queue"]):
            node_group = re.match("^xp([0-9]+?)mc([0-9]+?)$", arguments["--queue"])
            conf["cpu"] = int(node_group.group(1))
            conf["mem"] = int(node_group.group(1)) * int(node_group.group(2)) * 0.8
        elif arguments["--queue"] == "single":
            conf["cpu"] = 1
        elif arguments["--queue"] == "small":
            conf["cpu"] = 8
    if arguments["--process"] is not None:
        conf["cpu"] = int(arguments["--process"])
    if arguments["--mem"] is not None:
        conf["mem"] = int(arguments["--mem"])

    # Parse input file
    parse_gaussian_input(file_lines, conf)
    test_config(conf)
    print("conf: ")
    print(conf)
    write_config(conf)

    # Write bsub file
    if conf["queue"] is not None:
        prepare_bsub(conf)

    # Not execute if --no-exec is specified
    if not arguments["--no-exec"]:
        exec_program(conf)
