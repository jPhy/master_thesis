import cPickle
import os

def initialize_db():
    "Run this function before first use of ``save_first_proposal``."
    if os.path.exists('./first_proposals/database.info'):
        raise IOError('database exists')
    with open("./first_proposals/database.info", 'w') as dbfile:
        cPickle.dump([], dbfile)

def save_first_proposal(proposal, creation_parameters):
    """Save a ``proposal`` to "./first_proposals/i.prop" where i is the
    first free integer. Add attribute creation_parameters to ``proposal``.
    Save ``creation_parameters`` to "./first_proposals/database.info"

    :param proposal:

        The proposal to be saved

    :param creation_parameters:

        dict; the additional parameters that led to the ``proposal``

    """
    # add properties to ``proposal``
    proposal.creation_parameters = creation_parameters

    # find out filename for proposal
    i = 0
    while os.path.exists('./first_proposals/'+ str(i) + '.prop'):
        i += 1
    propfilename = './first_proposals/'+ str(i) + '.prop'

    # write the proposal to file
    with open(propfilename, 'w') as propfile:
        cPickle.dump(proposal, propfile)

    # update database
    with open("./first_proposals/database.info", 'r') as dbfile:
        db = cPickle.load(dbfile)
    db.append(creation_parameters)
    with open("./first_proposals/database.info", 'w') as dbfile:
        cPickle.dump(db, dbfile)

def load_first_proposal(i):
    """Load proposal number ``i`` saved by :py:func:`.save_first_proposal`.
    The proposal is read off the file "./first_proposals/i.prop"
    Can also parse negative ``i`` (standard python interpretation of
    negative numbers).

    """
    if i < 0:
        # find out filename for proposal (standard python interpretation of negative numbers)
        j = 0
        while os.path.exists('./first_proposals/'+ str(j) + '.prop'):
            j += 1
        i = j + i
    propfilename = './first_proposals/'+ str(i) + '.prop'
    propfilename = './first_proposals/'+ str(i) + '.prop'
    with open(propfilename, 'r') as propfile:
        proposal = cPickle.load(propfile)
    return proposal

def read_database():
    """Read the database constructed by calls to
    :py:func:`.save_first_proposal` (i.e. from file
    "./first_proposals/database.info").
    Return a list with dicts of the proposal parameters in the database.

    """
    with open("./first_proposals/database.info", 'r') as dbfile:
        db = cPickle.load(dbfile)
    return db
