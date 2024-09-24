from sacred import Experiment
from observer import create_observer 
from sacred.observers import MongoObserver
ex = Experiment('sample_experiment')

mongo_observer = create_observer()

# Create MongoObserver and append it to ex.observers
ex.observers.append(mongo_observer)


@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient

# The automain function needs to be at the end of the file.
# Otherwise everything below it is not defined yet when the experiment is run.
# The ex.automain will automatically be executed if you execute the file
@ex.automain
def my_main(message, recipient):
    ex.log_scalar('recipient', recipient)
    ex.log_scalar('message', message)
    print(message)
    return len(message)