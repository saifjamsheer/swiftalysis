### Taylor Swift Lyrics Generator

Generating lyrics using three methods: 
1. Self-built character-level LSTM text-generation model.
2. Self-built word-level LSTM & GRU text-generation model.
3. Retraining OpenAI's GPT-2 text-generating model on the scraped Taylor Swift lyrics.

> Full implementation steps will be listed once completed.

#### Character-Level Generator

Lyrics generated after 10 epochs with input 'The more I think about it now the less I know':
```
500 characters:

[Chorus]
I have that I'm forever
And you want you there what you're the one
I want you so out of it's for my first of this time
Wonded that the whole was the waiting gona
We know I want you

[Bridge]
And they say you said to breathe want

[Bridge]
We forget through the finds on the time
We'll see a fan you story when I'm one up to the night
And then I should wart so one
And there were spent, love a break and tall
Come back to be to make me
I got a gaine
But I don't want you love a bad to fall
```

#### Word-Level Generator

##### LSTM Model
Lyrics generated after 10 epochs with input 'Was she worth this':
```
50 words:

Was she worth this 
No no no no 
And no one's 
I know you taking my time 
And it's choice 

[Chorus] 
Come back come back come back to me like 
You wish it would 
Don't you think I was too young 
You shoulda known
```

##### GRU Model
Lyrics generated after 10 epochs with input 'Can't breathe whenever you're gone':
```
50 words:

Can't breathe whenever you're gone 
Can't go back no I forgot that you 

[Verse 1] 
Uh-uh things I turned out ooh-ooh-ooh-ooh 
I can see the way 
He was the way you take the way 
Of the world that comes a big Both] 
And all I had something
```

#### Remarks

This is a relatively simple model for lyric generation, and many changes can be made to improve performance.

Several takeaways from this project:
1. After around 15 epochs, the word-level generator began to simply reproduce lyrics from the corpus, indicating that it is very easy to overfit the model.
2. The character-level generator produced more novel sentences than the word-level generator.
3. An embedding layer may make the models more robust.
4. Splitting the dataset into chorus, verse, bridge, etc. and generating each section separately may produce more cohesive lyrics. 
