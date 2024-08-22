// Player


s.boot;
("fdbk_synthdef.scd").loadRelative;


	fund = 440,
	amp_ratio = 2,
	fm_ratio = 0.1,
	fm_index = 1,
	pnoise_amp = 0.5,
	lpf_freq = 5000,
	hpf_freq = 100,
	osc_fm_ratio = 0.5,

	pan = 0,
	input = 0,
	dry_env = 1,
	global_amp = 1,
	verb_mix = 0,

	lag_t = 0,
	delay_t = 0



// The two synthesizers
(
~leftSynth = Synth.new(\fdbk, [\pan, -1, \input, 0, \dry_env, 0, \verb_mix, 0.5]);
~rightSynth = Synth.new(\fdbk, [\pan, 1, \input, 1, \dry_env, 0, \verb_mix, 0.5]);
);


// The OSC receivers
(
OSCdef('\channel0', {
	arg msg;

	msg.postln;

	~leftSynth.set(
		\fund, msg[1],
	    \amp_ratio, msg[2],
		\fm_ratio, msg[3],
		\fm_index, msg[4],
		\pnoise_amp, msg[5],
		\lpf_freq, msg[6],
		\hpf_freq, msg[7],
		\osc_fm_ratio, msg[8]);
}, '/channel_0')
);

(
OSCdef('\channel1', {
	arg msg;

	msg.postln;

	~rightSynth.set(
		\fund, msg[1],
	    \amp_ratio, msg[2],
		\fm_ratio, msg[3],
		\fm_index, msg[4],
		\pnoise_amp, msg[5],
		\lpf_freq, msg[6],
		\hpf_freq, msg[7],
		\osc_fm_ratio, msg[8]);
}, '/channel_1')
)

