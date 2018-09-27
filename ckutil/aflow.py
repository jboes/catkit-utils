import numpy as np
import subprocess
import json
import os


def aflow_command(input_file, cmd='--sgdata', magmoms=None, extra=''):
    """Run an arbitrary AFLOW command from the terminal.

    Parameters:
    -----------
    input_file : str
        Path to Vasp POSCAR file to be read into AFLOW.
    cmd : str
        AFLOW command to be executed. `aflow --readme=symmetry` for all
        options, but `--sgdata`. `--edata`, and `--aflow-sym` are intended.
    magmoms : list of float
        Magnetic moments to be used for symmetry analysis. Must be in the
        same order of the input file.
    extra : str
        Additional arguments to be passed to the executable.

    Returns:
    --------
    output : str
        JSON output of the requested function call.
    """
    command = ['aflow', cmd, extra]

    if magmoms is not None:
        magmoms = ','.join(np.asarray(magmoms, dtype=str))
        command += ['--magmom=[{}]'.format(magmoms)]

    fpath = os.path.abspath(input_file)
    command += ['--print=json', '<', fpath]

    output = subprocess.check_output(' '.join(command), shell=True)
    output = json.loads(output)

    return output


def get_prototype_tag(atoms):
    """Return a prototype tag for a given bulk structure from AFLOW-SYM.
    This will automatically pass magnetic moments from the initial state of
    an atoms object.

    Currently only supports bulk structure classification. The tag is the
    alphabetically sorted chemical composition concatenated with the spacegroup
    and than the Wyckoff positions sorted alphabetically and then by
    multiplicity.

    This tag is not perfectly unique across all bulk prototypes due to a
    non-uniform sort order of the compositions. i.e. a matching tag guarantees
    an identical structure, but a non-match does not guarantee a unique
    structure. Also note that differences in magnetism can cause otherwise
    identical atoms to be symmetrically inequivalent.

    Parameters:
    -----------
    atoms : Atoms object
        Structure to determine the prototype tag of.

    Returns:
    ---------
    tag : str
        Single string representation of a bulk structure.
    """
    symbols = atoms.get_chemical_symbols()
    atoms = atoms[np.argsort(symbols)]

    positions = atoms.positions
    magmoms = atoms.get_initial_magnetic_moments()
    if np.all(magmoms == magmoms[0], axis=0):
        magmoms = None

    images, tags = [], []
    for p in positions:
        atoms.translate(-p)
        atoms.wrap()
        images += [atoms.copy()]
        atoms.write('POSCAR')

        output = aflow_command('POSCAR', magmoms=magmoms)

        sg = output['space_group_number']
        wc = output['Wyckoff_positions']
        sym = output['wyccar']['title'].split('|')[0].split()

        nn, nm, ns = [], [], []
        for entry in wc:
            if entry['name'] in nn:
                i = nn.index(entry['name'])
                nm[i] += entry['multiplicity']
                ns[i] = ''.join(sorted(ns[i] + entry['Wyckoff_letter']))
            else:
                nn += [entry['name']]
                nm += [entry['multiplicity']]
                ns += [entry['Wyckoff_letter']]
        srt = np.lexsort([ns, nm])
        tags += [sg + '_' + '_'.join(np.array(ns)[srt]) + '_'
                 + '_'.join(np.array(sym)[srt])]
    os.unlink('POSCAR')

    selection = np.argsort(tags)[0]
    atoms = images[selection]
    tag = tags[selection]

    return tag, atoms
