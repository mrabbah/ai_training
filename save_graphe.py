import tensorflow as tf

is_simulation = tf.placeholder(dtype=tf.bool, 'is_simulation')

lat = tf.placeholder(dtype=tf.float32, 'latitude')
lng = tf.placeholder(dtype=tf.float32, 'longitude')
target_lat = tf.placeholder(dtype=tf.float32, 'target_latitude')
target_lng = tf.placeholder(dtype=tf.float32, 'target_longitude')

dist_sim = tf.sqrt(tf.pow(lat - target_lat, 2) + tf.pow(lng - target_lng, 2), 'simulator_distance')

R = 6371e3; # metres

#TODO https://www.movable-type.co.uk/scripts/latlong.html
dist_haversine = tf.atan2(lat, lng, 'haversine_distance')

dist = tf.cond(is_simulation, lambda: dist_sim, lambda: dist_haversine, 'distance')

observation_size = 4

vector_in = tf.placeholder(shape=[None, observation_size], dtype=tf.float32, 
							name='vector_observation')

normalized_state = tf.nn.sigmoid(vector_in, 'normalized_state')

init = tf.global_variables_initializer()

sess = tf.Session()

fw = tf.summary.FileWriter('graph', sess.graph)

sess.run(init)

print(sess.run(normalized_state, {vector_in : [[10, 0, -5, -1], [2, 0, 100, 1.5], [10, 0, 3, 0], [-100, 0, -5, 1]]}))

sess.close()
