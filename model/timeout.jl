struct TimeoutException <: Exception end

function timeout(f, n, message="")
    f_task = @async try
        f()
    catch err
        err isa TimeoutException || rethrow()
        :timeout
    end
    interrupt_task = @async begin
        sleep(n)
        if f_task.state != :done
            @warn "Timeout! $message"
            Base.throwto(f_task, TimeoutException())
        end
    end
    result = fetch(f_task)
    if result == :timeout
        throw(TimeoutException())
    end
    result
end