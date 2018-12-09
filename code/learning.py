import os, csv
import tensorflow as tf
from model import FeedModel
import experience_batcher as experbatcher
from experience_collector import ExperienceCollector
import play
NUM_OF_ACTIONS = 15

def make_run_inference(session, model):
  """Make run_inference() function for given session and model."""

  def run_inference(state_batch):
    """Run inference on a given state_batch. Returns a q value batch."""
    return session.run(model.q_values,
                       feed_dict={model.state_batch_placeholder: state_batch})
  return run_inference


def make_get_q_values(session, model):
  """Make get_q_values() function for given session and model."""

  run_inference = make_run_inference(session, model)
  def get_q_values(state):
    """Run inference on a single (1, 3) state matrix."""
    state_vector = state.flatten()
    state_batch = np.array([state_vector])
    q_values_batch = run_inference(state_batch)
    return q_values_batch[0]
  return get_q_values

def run_training(train_dir):
    with tf.Graph().as_default():
        model = FeedModel()
        saver = tf.train.Saver()
        session = tf.Session()
        summary_writer = tf.summary.FileWriter(train_dir,
                                               graph_def=session.graph_def,
                                               flush_secs=10)
        #resume = os.path.exists(train_dir)
        #print("Resume: ", resume)
        #if resume:
        #    print("Resuming: ", train_dir)
        #    saver.restore(session, tf.train.latest_checkpoint(train_dir))
        #else:
        print("Starting new training: ", train_dir)
        session.run(model.init)
    print("Aravind, tell me you are here")
    run_inference = make_run_inference (session, model)
    print ("Before for loop in learning.py 1")
    get_q_values = make_get_q_values (session, model)
    STATE_NORMALIZE_FACTOR = 1
    print ("Before for loop in learning.py 2")
    experience_collector = ExperienceCollector ()
    print ("Before for loop in learning.py 3")
    batcher = experbatcher.ExperienceBatcher (experience_collector, run_inference, get_q_values, STATE_NORMALIZE_FACTOR)
    print ("Before for loop in learning.py 4")
    test_experiences = experience_collector.collect (play.random_strategy, NUM_OF_ACTIONS)
    print("Before for loop in learning.py 5")
    for state_batch, targets, actions in batcher.get_batches_stepwise ():

        global_step, _ = session.run ([model.global_step, model.train_op],
                                    feed_dict={model.state_batch_placeholder: state_batch,
                                      model.targets_placeholder: targets, model.actions_placeholder: actions, })
        if global_step % 1e3 == 0 and global_step != 0:
            saver.save (session, train_dir + "/checkpoint", global_step=global_step)
            loss = write_summaries (session, batcher, model, test_experiences, summary_writer)
            print ("Step:", global_step, "Loss:", loss)

def main(args):
    if len(args) != 2:
        print("Usage: %s train_dir" % args[0])
        sys.exit(1)

    run_training('/home/ece.ecoprt/Reinforcement learning/Dynamic_Vehicle_Control_using_RL-master/code_aravind/Dynamic_Vehicle_Control_using_RL/code/data')


if __name__ == '__main__':
    tf.app.run()
