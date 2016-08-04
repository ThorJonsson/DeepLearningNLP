-- Row-normalize a matrix
function normalize(m)
  m_norm = torch.zeros(m:size())
  for i = 1, m:size(1) do
    m_norm[i] = m[i] / torch.norm(m[i])
  end
  return m_norm
end

function get_word_embeddings(checkpoint)
  return checkpoint.protos.rnn.modules[2].weight:double()
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function get_sim_words(w, k, checkpoint, word2idx, idx2word)
    word_vecs = get_word_embeddings(checkpoint)
    if word_vecs_norm == nil then
        word_vecs_norm = normalize(word_vecs)
    end
    if type(w) == "string" then
        if word2idx[w] == nil then
           print("'"..w.."' does not exist in vocabulary.")
           return nil
        else
            w = word_vecs_norm[word2idx[w]]
        end
    end
    -- Compute the matrix vector product
    local sim = torch.mv(word_vecs_norm, w)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {idx2word[idx[i]], -sim[i]}
    end
    return r
end



---print similar words
function print_sim_words(words, k, checkpoint, word2idx, idx2word)
    for i = 1, #words do
        r = get_sim_words(words[i], k, checkpoint, word2idx, idx2word)
        if r ~= nil then
            print("-------"..words[i].."-------")
            for j = 1, k do
                print(string.format("%s, %.4f", r[j][1], r[j][2]))
            end
        end
    end
end


function forward_update_states(prev_word, gmod)
    --dbg()
    local lst = gmod.rnn:forward{prev_word, unpack(current_state)}
    -- lst is a list of [state1,state2,..,stateN,output]. 
    -- The output is the log probabilities, i.e. element #lst
    -- Now lst contains current state of the nn
    local current_state = {} -- reset
    for i=1, state_size do table.insert(current_state, lst[i]) end -- clone
    return lst[#lst]
end
